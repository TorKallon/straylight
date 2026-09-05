#!/usr/bin/env node

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { appendFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { z } from "zod/v4";

import { type ApiResponse, BrunnApiClient, BrunnApiError } from "./api-client.js";
import { registerMessagingTools } from "./messaging-tools.js";
import { compactReasoningResponse } from "./reasoning-view.js";

const reference = z.string().min(1);
const entryReference = z.string()
  .regex(/^entry:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i)
  .describe("Exact entry:... reference copied from a Brunn response; never infer or invent one.");
const assetReference = z.string()
  .regex(/^entry:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i)
  .describe("Exact entry:... binary reference copied from a Brunn response.");
const jsonObject = z.record(z.string(), z.unknown());
const MAX_CHECKPOINT_BYTES = 4 * 1024 * 1024;
const MAX_CHECKPOINT_ITEMS = 4_096;
const checkpointIdentityReference = printableUtf8String(256);
const checkpointStateReference = printableUtf8String(4_096);
const checkpointSourceReference = printableUtf8String(4_096).describe(
  "An exact entry:... reference or relative Markdown path returned by search/read.",
);
const checkpointIdempotencyKey = z.string().min(1).max(256).refine(
  (value) => Buffer.byteLength(value, "utf8") <= 256
    && !/[\u0000-\u001f\u007f-\u009f]/u.test(value),
  "idempotency_key must contain at most 256 UTF-8 bytes and no control characters",
).describe(
  "Stable replay identity. Reuse this exact key with an identical checkpoint payload after an ambiguous outcome.",
);
const checkpointText = z.string().max(MAX_CHECKPOINT_BYTES).refine(
  (value) => Buffer.byteLength(value, "utf8") <= MAX_CHECKPOINT_BYTES,
  "checkpoint strings are limited to 4 MiB of UTF-8 text",
);
const checkpointStructuredItem = jsonObject.refine(
  (value) => serializedUtf8Length(value) <= MAX_CHECKPOINT_BYTES,
  "structured checkpoint items are limited to 4 MiB of serialized UTF-8 JSON",
);
const checkpointState = z.object({
  objective: checkpointText.min(1),
  project: z.string()
    .min(1)
    .max(100)
    .regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/u)
    .optional()
    .describe("Optional registered Brunn project slug for durable checkpoint linkage."),
  current_state: z.union([
    checkpointText,
    z.array(checkpointText).max(MAX_CHECKPOINT_ITEMS),
  ]).optional(),
  decisions: z.array(checkpointText).max(MAX_CHECKPOINT_ITEMS).optional(),
  open_questions: z.array(checkpointText).max(MAX_CHECKPOINT_ITEMS).optional(),
  next_actions: z.array(checkpointText).max(MAX_CHECKPOINT_ITEMS).optional(),
  artifacts: z.array(checkpointText).max(MAX_CHECKPOINT_ITEMS).optional(),
  ordered_goals: z.array(
    z.union([checkpointText, checkpointStructuredItem]),
  ).max(MAX_CHECKPOINT_ITEMS).optional(),
  state_refs: z.array(checkpointStateReference).max(MAX_CHECKPOINT_ITEMS).optional(),
  acceptance_gates: z.array(
    z.union([checkpointText, checkpointStructuredItem]),
  ).max(MAX_CHECKPOINT_ITEMS).optional(),
}).refine(
  (value) => serializedUtf8Length(value) <= MAX_CHECKPOINT_BYTES,
  "checkpoint state is limited to 4 MiB of serialized UTF-8 JSON",
);

const queryItem = z.object({
  id: z.string().optional(),
  goal: z.string().optional(),
  query: z.string().min(1),
  modes: z.array(
    z.enum(["exact", "lexical", "semantic"]),
  ).optional().describe(
    "Omit for hybrid search. Use exact for a literal path or title, lexical for words, or semantic for meaning.",
  ),
  limit: z.number().int().min(1).max(50).default(8),
});

const editionDate = z.string().regex(/^\d{4}-\d{2}-\d{2}$/).describe(
  "Exact edition date YYYY-MM-DD.",
);
const secretName = z.string().min(1).max(120).regex(/^[a-z0-9][a-z0-9._-]{0,119}$/i).describe(
  "Stable secret name such as datadog-prod-api-key. Case-insensitive; stored lowercase.",
);
const storyKey = z.string().regex(/^[a-z0-9][a-z0-9-]{2,79}$/);
const storyUrl = z.string().min(1).max(2_048);
const documentSlug = z.string()
  .regex(/^[a-z0-9](?:[a-z0-9-]{0,78}[a-z0-9])$/)
  .describe("Stable lowercase document slug, 2 to 80 characters; reuse it to revise the same document.");
const documentSourceLabel = z.string().min(1).max(240);
const documentSource = z.union([
  z.object({
    label: documentSourceLabel,
    entry_ref: entryReference,
  }).strict(),
  z.object({
    label: documentSourceLabel,
    url: z.string().min(1).max(2_048).refine(isSafeDocumentSourceUrl, {
      message: "document source URL must be HTTP(S) and must not contain credentials",
    }),
  }).strict(),
]);

function isSafeDocumentSourceUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return (url.protocol === "http:" || url.protocol === "https:")
      && url.username === ""
      && url.password === "";
  } catch {
    return false;
  }
}

const briefingStoryRef = z.object({
  key: storyKey.describe(
    "Lowercase story slug. When briefing.dedupe returned this story, copy its story_key " +
    "verbatim; never invent a variant of a key the ledger already has. Mint a new slug only " +
    "for a story with no dedupe match.",
  ),
  urls: z.array(storyUrl).max(8).optional().describe(
    "Canonical source URLs for the story; the service canonicalizes and hashes them for dedupe.",
  ),
  title: z.string().max(500).optional(),
  entities: z.array(z.string().max(120)).max(16).optional(),
  event_at: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).optional().describe(
    "Exact date the underlying event happened, YYYY-MM-DD. Omit this field when unknown; never guess.",
  ),
});

const briefingTimes = z.object({
  published_at: z.string().optional(),
  event_at: z.string().optional(),
  first_seen_at: z.string().optional(),
});

const briefingItem = z.object({
  id: z.string().regex(/^[a-z0-9][a-z0-9-]{1,63}$/).describe(
    "Lowercase item slug, unique within the edition. Reuse the same id when republishing an " +
    "unchanged or revised item so revision deltas stay accurate.",
  ),
  kind: z.enum(["news", "metric", "health", "ops", "digest", "tracker", "schedule"]),
  headline_md: z.string().max(500),
  body_md: z.string().max(4_000).optional(),
  why_it_matters: z.string().max(1_000).optional(),
  detail_md: z.string().max(16_000).optional(),
  what_changed: z.string().max(1_000).optional(),
  delta: z.enum(["new", "update", "corroboration"]).optional().describe(
    "Omit this field for a first delivery; the service records new. Use update or corroboration " +
    "only when briefing.dedupe showed the story was already delivered.",
  ),
  story: briefingStoryRef.optional(),
  times: briefingTimes.optional(),
});

const briefingSection = z.object({
  topic: z.string().max(80).describe("Exact topic slug from briefing.topics; never invent one."),
  title: z.string().max(200),
  items: z.array(briefingItem).max(32),
});

const briefingOmission = z.object({
  story_key: storyKey.optional().describe(
    "Story key for the omitted story; copy it verbatim from the briefing.dedupe result that " +
    "identified the duplicate when one exists.",
  ),
  urls: z.array(storyUrl).max(8).optional(),
  reason: z.string().min(1).max(1_000),
});

const dedupeCandidate = z.object({
  urls: z.array(storyUrl).max(8).optional(),
  title: z.string().max(500).optional(),
  summary: z.string().max(4_000).optional(),
  event_at: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).optional().describe(
    "Exact date the underlying event happened, YYYY-MM-DD. Omit this field when unknown; never guess.",
  ),
  topic: z.string().max(80).optional(),
  story_key: storyKey.optional().describe(
    "Story key to look up exactly. Copy keys verbatim from prior briefing.dedupe or " +
    "briefing.publish results when checking a known story; a key absent from the ledger " +
    "simply returns no match.",
  ),
});

const notificationSource = z.object({
  type: z.string().min(1).max(64),
  ref: z.string().min(1).max(500),
  version_ref: z.string().min(1).max(500).optional(),
});

const notificationTarget = z.discriminatedUnion("type", [
  z.object({ type: z.literal("notification") }),
  z.object({ type: z.literal("today") }),
  z.object({
    type: z.literal("briefing"),
    date: editionDate,
    edition: z.string().min(1).max(64),
    item_id: z.string().min(1).max(200).optional(),
  }),
  z.object({
    type: z.literal("entry"),
    entry_ref: z.string().min(1).max(500).describe(
      "Exact entry:... reference returned by Brunn; never infer one from a title or path.",
    ),
  }),
]);

const taskRef = z.string()
  .regex(/^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/u)
  .describe(
    "Raw lowercase UUIDv7 returned as task_ref. Do not prefix it with task: and never infer one.",
  );
const taskSlug = z.string()
  .min(1)
  .max(100)
  .regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/u);
const contextSlug = z.string()
  .min(1)
  .max(80)
  .regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/u);
const surfaceSlug = z.string()
  .min(1)
  .max(64)
  .regex(/^[a-z][a-z0-9._-]{0,63}$/u);
const localDate = z.iso.date().describe("Local calendar date in YYYY-MM-DD form.");
const rfc3339Timestamp = z.iso.datetime({ offset: true }).describe(
  "Exact RFC3339 timestamp with an offset or Z suffix.",
);
const taskIdempotencyKey = printableUtf8String(240).describe(
  "Stable durable replay identity. Reuse the exact key only with the identical mutation payload.",
);
const taskWriteSource = z.string()
  .regex(/^(?:owner|agent:[a-zA-Z0-9][a-zA-Z0-9._:-]{0,199})$/u)
  .describe(
    "Use owner only for a value the owner directly supplied; agent:<id> for inference. "
    + "todoist and derived are reserved for internal service writers.",
  );
const taskCompletedVia = z.union([
  z.literal("ios"),
  z.literal("web"),
  z.string().regex(/^agent:[a-zA-Z0-9][a-zA-Z0-9._:-]{0,199}$/u),
]).describe("Actual completion surface. MCP agents normally use agent:<id>.");
const contextList = z.array(contextSlug).max(20).refine(
  (values) => new Set(values).size === values.length,
  "contexts must not contain duplicates",
);
const surfaceDefaults = z.record(surfaceSlug, contextList).refine(
  (values) => Object.keys(values).length <= 20,
  "surface_defaults accepts at most 20 surfaces",
);

function sourcedTaskCell<Value extends z.ZodType>(value: Value) {
  return z.object({
    value,
    source: taskWriteSource,
    note: z.string().min(1).max(1_000).optional(),
  }).strict();
}

const costOfDelay = z.union([
  z.object({
    amount_cents: z.number().int().nonnegative(),
    per: z.enum(["day", "week", "month"]),
    since: localDate,
    note: z.string().min(1).max(1_000).optional(),
  }).strict(),
  z.object({
    flag: z.literal(true),
    since: localDate,
    note: z.string().min(1).max(1_000).optional(),
  }).strict(),
]);

const taskCaptureItem = z.object({
  client_ref: printableUtf8String(200).optional().describe(
    "Caller-local correlation value echoed in the capture result; not a task identifier.",
  ),
  raw_text: z.string().min(1).max(20_000).describe(
    "The owner's original task sentence, preserved as the capture basis.",
  ),
  title: z.string().min(1).max(500).optional(),
  notes: sourcedTaskCell(z.string().max(20_000).nullable()).optional(),
  project: sourcedTaskCell(taskSlug).optional(),
  ready_at: sourcedTaskCell(rfc3339Timestamp.nullable()).optional(),
  soft_due: sourcedTaskCell(localDate.nullable()).optional(),
  hard_due: sourcedTaskCell(rfc3339Timestamp.nullable()).optional(),
  hard_due_lead_days: sourcedTaskCell(z.number().int().min(0).max(3_650).nullable()).optional(),
  cost_of_delay: sourcedTaskCell(costOfDelay.nullable()).optional(),
  required_contexts: sourcedTaskCell(contextList).optional(),
  estimate_minutes: sourcedTaskCell(z.number().int().min(1).max(10_080).nullable()).optional(),
  captured_from: printableUtf8String(4_096).optional().describe(
    "Exact conversation or entry reference supporting this capture; omit when none was supplied.",
  ),
}).strict();

const correctionAuditFields = {
  source: taskWriteSource,
  note: z.string().min(1).max(1_000).optional(),
  reason: z.string().min(1).max(1_000).optional().describe(
    "Why this correction supersedes the previous value; retained in the corrections log.",
  ),
};

const taskUpdateOperation = z.union([
  z.object({
    type: z.literal("correct"),
    field: z.enum([
      "title",
      "notes",
      "project",
      "ready_at",
      "soft_due",
      "hard_due",
      "hard_due_lead_days",
      "cost_of_delay",
      "estimate_minutes",
      "recurrence",
    ]),
    value: z.unknown(),
    ...correctionAuditFields,
  }).strict(),
  z.object({
    type: z.literal("correct"),
    field: z.literal("required_contexts"),
    value: contextList,
    ...correctionAuditFields,
  }).strict(),
  z.object({
    type: z.literal("complete"),
    source: taskWriteSource,
    completed_via: taskCompletedVia,
  }).strict(),
  z.object({ type: z.literal("reopen"), source: taskWriteSource }).strict(),
  z.object({
    type: z.literal("snooze"),
    until: rfc3339Timestamp,
    source: taskWriteSource,
  }).strict(),
  z.object({
    type: z.literal("snooze"),
    days: z.number().int().min(1).max(3_650),
    source: taskWriteSource,
  }).strict(),
  z.object({
    type: z.literal("drop"),
    reason: z.string().min(1).max(1_000).optional(),
    source: taskWriteSource,
  }).strict(),
  z.object({
    type: z.literal("wait_on"),
    who_or_what: z.string().min(1).max(1_000),
    check_back_at: rfc3339Timestamp.optional(),
    source: taskWriteSource,
  }).strict(),
  z.object({ type: z.literal("unpark"), source: taskWriteSource }).strict(),
  z.object({ type: z.literal("pin_today"), source: taskWriteSource }).strict(),
  z.object({ type: z.literal("unpin"), source: taskWriteSource }).strict(),
  z.object({ type: z.literal("confirm_hard"), source: taskWriteSource }).strict(),
  z.object({ type: z.literal("downgrade_to_soft"), source: taskWriteSource }).strict(),
]);

const contextOperation = z.union([
  z.object({
    type: z.literal("list"),
    include_archived: z.boolean().default(false),
    limit: z.number().int().min(1).max(100).default(50),
    cursor: contextSlug.optional(),
  }).strict(),
  z.object({
    type: z.literal("create"),
    slug: contextSlug.optional().describe(
      "Optional canonical lowercase-kebab slug. Omit it to derive the slug from display_name.",
    ),
    display_name: z.string().min(1).max(120),
    aliases: z.array(z.string().min(1).max(120)).max(32).optional(),
    description: z.string().min(1).max(1_000).optional(),
    source: taskWriteSource,
    confirm_new: z.boolean().default(false).describe(
      "Leave false initially. Set true only after Brunn returns suggested_existing and the owner confirms a new context.",
    ),
    idempotency_key: taskIdempotencyKey,
  }).strict(),
  z.object({
    type: z.literal("merge"),
    from: contextSlug,
    into: contextSlug,
    expected_from_version: z.number().int().positive(),
    expected_into_version: z.number().int().positive(),
    source: taskWriteSource,
    reason: z.string().min(1).max(1_000).optional(),
    idempotency_key: taskIdempotencyKey,
  }).strict().refine((value) => value.from !== value.into, {
    message: "merge source and destination must differ",
  }),
  z.object({
    type: z.literal("archive"),
    slug: contextSlug,
    archived: z.boolean().default(true),
    expected_version: z.number().int().positive(),
    source: taskWriteSource,
    idempotency_key: taskIdempotencyKey,
  }).strict(),
  z.object({
    type: z.literal("set_available"),
    surface: surfaceSlug,
    contexts_available: contextList,
    expected_version: z.number().int().nonnegative().describe(
      "Use 0 only when creating defaults for an unseeded surface; otherwise use the positive surface_defaults version returned by list.",
    ),
    source: taskWriteSource,
    idempotency_key: taskIdempotencyKey,
  }).strict(),
]);

const taskSettingsOperation = z.union([
  z.object({ type: z.literal("get") }).strict(),
  z.object({
    type: z.literal("update"),
    expected_version: z.number().int().positive(),
    idempotency_key: taskIdempotencyKey,
    timezone: z.string().min(1).max(80).optional(),
    hard_lead_days: z.number().int().min(1).max(90).optional(),
    hard_second_lead_hours: z.number().int().min(1).max(2_160).optional(),
    due_day_local_time: z.string().regex(/^(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$/u).optional(),
    soft_window_days: z.number().int().min(1).max(90).optional(),
    triage_after_days: z.number().int().min(1).max(3_650).optional(),
    waiting_followup_days: z.number().int().min(1).max(3_650).optional(),
    quiet_override_enabled: z.boolean().optional(),
    quiet_override_within_hours: z.number().int().min(1).max(168).optional(),
    quiet_hours_start: z.string()
      .regex(/^(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$/u)
      .optional(),
    quiet_hours_end: z.string()
      .regex(/^(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$/u)
      .optional(),
    surface_defaults: surfaceDefaults.optional(),
  }).strict().refine(
    (value) => Object.keys(value).some(
      (key) => !["type", "expected_version", "idempotency_key"].includes(key),
    ),
    { message: "settings update requires at least one changed setting" },
  ),
]);

function createReadItem(maxChars: number) {
  return z.object({
    ref: reference.optional().describe(
      "Exact record reference copied verbatim from a Brunn State response. Never infer or invent a reference.",
    ),
    path: z.string().min(1).optional().describe(
      "Exact source path copied verbatim from a Brunn State response. Never synthesize a filename from a title or topic.",
    ),
    view: z.enum([
      "current_state",
      "current_truth",
      "outline",
      "full",
      "range",
    ]).optional(),
    start: z.number().int().min(1).optional(),
    end: z.number().int().min(1).optional(),
    max_chars: z.number().int().min(1).max(maxChars).optional(),
  }).refine((value) => value.ref !== undefined || value.path !== undefined, {
    message: "read request requires ref or path",
  });
}

export interface BrunnMcpServerOptions {
  surface?: "local" | "remote";
  includeStructuredContent?: boolean;
  messagingEnabled?: boolean;
  maxReadChars?: number;
}

export function createBrunnMcpServer(
  client: BrunnApiClient,
  options: BrunnMcpServerOptions = {},
): McpServer {
  const surface = options.surface ?? "local";
  const includeStructuredContent = options.includeStructuredContent
    ?? process.env.BRUNN_MCP_INCLUDE_STRUCTURED_CONTENT === "1";
  const messagingEnabled = options.messagingEnabled
    ?? process.env.BRUNN_MESSAGING_ENABLED === "true";
  const maxReadChars = options.maxReadChars ?? (surface === "remote" ? 120_000 : 500_000);
  const server = new McpServer({
    name: "Brunn",
    version: "0.1.0",
  }, surface === "remote" ? {
    instructions:
      "Brunn is the durable context store. Start substantive work with memory.open for the actual task, " +
      "then use memory.query and memory.read only for relevant evidence. Persist source material with " +
      "memory.capture, durable current state or corrections with memory.write, and resumable work with " +
      "memory.checkpoint. If Brunn is unavailable, fail closed instead of inventing or substituting context.",
  } : {});

  function registerJsonTool<Shape extends z.ZodRawShape>(
    name: string,
    description: string,
    inputSchema: Shape,
    invoke: (input: z.infer<z.ZodObject<Shape>>) => Promise<ApiResponse>,
  ): void {
    registerJsonToolOnServer(
      server,
      includeStructuredContent,
      name,
      description,
      inputSchema,
      invoke,
    );
  }

registerJsonTool(
  "memory.open",
  "Open or resume the workspace and receive bounded, coherent source documents relevant to the task.",
  {
    task: z.string().min(1),
    hints: z.object({
      authorization_scope: z.string().optional(),
      root_refs: z.array(reference).optional(),
      open_object_refs: z.array(reference).optional(),
    }).optional(),
    resume_checkpoint_ref: reference.optional().describe(
      "Exact checkpoint:... reference supplied by the caller. Omit this field when no exact " +
      "checkpoint reference was supplied; never invent one or use placeholders such as latest.",
    ),
    token_budget: z.number().int().min(1).optional(),
    modes: z.array(
      z.enum(["exact", "lexical", "semantic"]),
    ).optional().describe(
      "Omit for the server policy. Evaluation arms may explicitly restrict open to exact and lexical retrieval.",
    ),
  },
  (input) => client.request("/v1/workspace/open", input),
);

registerJsonTool(
  "memory.query",
  "Search current workspace files by exact path or title, full text, and semantic similarity.",
  {
    session_id: reference,
    queries: z.array(queryItem).min(1).max(16),
    token_budget: z.number().int().min(1).optional().describe(
      "Optional total search response budget in tokens; the service converts it to a bounded character cap when budget-contracted retrieval is enabled.",
    ),
  },
  (input) => client.request("/v1/workspace/search", input),
);

registerJsonTool(
  "memory.read",
  "Batch exact reads of current Markdown files or checkpoints by returned entry reference or path.",
  {
    session_id: reference,
    requests: z.array(createReadItem(maxReadChars)).min(1).max(32),
  },
  (input) => client.request("/v1/workspace/read", input),
);

registerJsonTool(
  "memory.changes",
  "Page through workspace changes after an exact generation cursor.",
  {
    since_generation: z.number().int().nonnegative().default(0),
    limit: z.number().int().min(1).max(2_000).default(200),
  },
  (input) => client.workspaceChanges(input.since_generation, input.limit),
);

registerJsonTool(
  "memory.capture",
  "Persist ordinary source-backed context as a durable Markdown capture.",
  {
    content: z.string().min(1).max(256_000),
    source: z.object({
      ref: reference.optional(),
      external_ref: z.string().min(1).max(2_000).optional(),
      title: z.string().min(1).max(500).optional(),
      kind: z.string().min(1).max(120).optional(),
      origin: z.enum(["user", "external", "agent", "tool", "system"]).optional(),
      media_type: z.string().optional(),
      locator: jsonObject.optional(),
      metadata: jsonObject.optional(),
      content_hash: z.string().optional(),
    }).refine((value) => value.ref !== undefined || value.title !== undefined, {
      message: "capture source requires ref or title",
    }),
    intent: z.string().min(1).optional(),
    idempotency_key: z.string().min(1).max(240).optional(),
  },
  (input) => client.request("/v1/workspace/capture", input),
);

registerJsonTool(
  "memory.write",
  "Create or update one Markdown workspace file. Supply expected_version when preventing a stale overwrite matters. " +
  "If both expected_version and idempotency_key were supplied and a client or transport reports an ambiguous outcome, " +
  "retry at most once with every argument and the key unchanged; never mint a new key for that retry.",
  {
    path: z.string().min(1).max(1_024),
    content: z.string().max(4 * 1024 * 1024),
    media_type: z.enum(["text/markdown", "text/plain"]).default("text/markdown"),
    expected_version: z.number().int().nonnegative().optional().describe(
      "Optimistic version guard; zero means create. Pair it with idempotency_key for a replay-safe guarded retry.",
    ),
    idempotency_key: z.string().min(1).max(240).optional().describe(
      "Stable retry identity. Reuse it only with an identical payload; it does not replace expected_version.",
    ),
    metadata: jsonObject.optional(),
  },
  (input) => client.request("/v1/workspace/write", input),
);

registerJsonTool(
  "memory.checkpoint",
  "Write a deterministic checkpoint Markdown file with exact file/version/hash references and a workspace generation.",
  {
    session_id: checkpointIdentityReference,
    parent_checkpoint_id: checkpointIdentityReference.optional(),
    idempotency_key: checkpointIdempotencyKey,
    state: checkpointState,
    source_refs: z.array(checkpointSourceReference).max(64).optional(),
  },
  (input) => client.request("/v1/workspace/checkpoint", input),
);

if (surface === "local") {
  registerJsonTool(
    "memory.stage",
    "Upload binary files from the adapter's sandboxed import root without placing bytes in model context.",
    {
      scope: z.string().min(1),
      stable_import_id: z.string().min(1).max(240).optional(),
      describe_binaries: z.boolean().default(true).describe(
        "Generate searchable, explicitly non-authoritative descriptions for native files.",
      ),
      files: z.array(z.object({
        path: z.string().min(1).describe(
          "Path below BRUNN_MCP_IMPORT_ROOT; it is retained as the logical vault path unless name is supplied.",
        ),
        name: z.string().min(1).optional().describe(
          "Optional logical vault path override. This is not merely a basename.",
        ),
        media_type: z.string().optional(),
      })).min(1).max(32),
    },
    (input) => client.stage(
      input.scope,
      input.stable_import_id,
      input.files,
      input.describe_binaries,
    ),
  );
}

registerJsonTool(
  "document.publish",
  "Publish or revise a polished, human-facing Markdown document and return its direct Brunn links. " +
  "Use this when the user asks to show, open, or read a plan, document, detailed analysis, vacation " +
  "information, feature specification, or comparable long-form material; the request phrasing is the " +
  "publication trigger. Do not use it for routine replies, raw imports, internal evidence, or uncurated " +
  "files. Republishing the same slug revises the stable latest-document link. Return the response's stable " +
  "`url` field to the user instead of an entry reference; use `version_url` only when the user explicitly " +
  "asked for a pinned historical revision.",
  {
    slug: documentSlug,
    title: z.string().min(1).max(240),
    body_md: z.string().min(1).max(4 * 1024 * 1024).refine(
      (value) => value.trim().length > 0,
      { message: "body_md must contain non-whitespace Markdown" },
    ),
    summary: z.string().min(1).max(1_000).optional(),
    sources: z.array(documentSource).max(32).optional().describe(
      "Curated provenance for this human-facing document. Do not attach unrelated raw imports or internal evidence.",
    ),
    idempotency_key: z.string().min(1).max(240).optional(),
    expected_version: z.number().int().nonnegative().optional().describe(
      "Supply expected_version only when preventing a known stale overwrite matters; zero means create.",
    ),
  },
  (input) => client.request("/v1/workspace/documents/publish", input),
);

registerJsonTool(
  "document.get",
  "Retrieve one intentionally published human-facing Markdown document and its direct Brunn links. " +
  "Omit version for the stable latest document; request a positive version only for an explicit historical " +
  "revision. Return the response's stable `url` field by default instead of an entry reference; return " +
  "`version_url` only for an explicitly requested historical revision.",
  {
    slug: documentSlug,
    version: z.number().int().positive().optional().describe(
      "Optional historical document version. Omit it to retrieve the current version behind the stable link.",
    ),
  },
  (input) => {
    const query = input.version === undefined
      ? ""
      : `?${new URLSearchParams({ version: String(input.version) }).toString()}`;
    return client.request(
      `/v1/workspace/documents/${encodeURIComponent(input.slug)}${query}`,
    );
  },
);

registerJsonTool(
  "memory.status",
  "Inspect current service and dependency status.",
  {},
  () => client.request("/v1/status"),
);

registerJsonTool(
  "location.presence",
  "Read the owner's current formatted location presence. Returns no raw location reports.",
  {},
  async () => {
    try {
      return await client.request("GET", "/v1/location/presence");
    } catch (error) {
      const detail = error instanceof BrunnApiError ? error.body.error : undefined;
      if (
        error instanceof BrunnApiError &&
        error.status === 404 &&
        typeof detail === "object" &&
        detail !== null &&
        "code" in detail &&
        detail.code === "location_presence_not_found"
      ) {
        return { status: 200, body: { status: "none" }, elapsedMs: 0 };
      }
      throw error;
    }
  },
);

registerJsonTool(
  "location.rederive",
  "Rebuild derived location presence and visit month rows from retained raw reports for an optional bounded window. Returns counts only.",
  {
    from: rfc3339Timestamp.optional(),
    to: rfc3339Timestamp.optional(),
  },
  (input) => client.request("POST", "/v1/location/rederive", input),
);

registerJsonTool(
  "asset.upload_url",
  "Authorize one binary upload to an exact workspace path (same permission as memory.write). " +
    "PUT the raw file bytes to put_url using the returned headers before expires_at. " +
    "Never put file bytes or base64 in a tool call. Read the PUT JSON result and reference its entry_ref/path. " +
    "An existing path requires its current expected_version; omission means create only. " +
    "Retry with the SAME returned permission after an uncertain PUT; HTTP 409 upload_completed contains the published result.",
  {
    path: z.string().min(1).max(1024),
    media_type: z.string().min(1).max(255),
    size_bytes: z.number().int().nonnegative().max(4 * 1024 * 1024 * 1024),
    sha256: z.string().regex(/^(?:sha256:)?[0-9a-f]{64}$/i).optional(),
    expected_version: z.number().int().nonnegative().optional(),
  },
  (input) => client.request("POST", "/v1/uploads", input),
);

registerJsonTool(
  "asset.list",
  "List current binary workspace entries and their exact hashes, versions, sizes, and description metadata.",
  {
    session_id: reference.describe(
      "Exact session:... reference returned by memory.open.",
    ),
    offset: z.number().int().nonnegative().default(0),
    limit: z.number().int().min(1).max(500).default(100),
  },
  (input) => client.listAssets(input.session_id, input.offset, input.limit),
);

registerJsonTool(
  "asset.metadata",
  "Read metadata for one exact binary workspace entry and optional historical version without downloading bytes.",
  {
    session_id: reference.describe(
      "Exact session:... reference returned by memory.open and retained for workspace continuity.",
    ),
    asset_ref: assetReference,
    version: z.number().int().positive().optional().describe(
      "Optional exact historical version. Omit it to read the current version.",
    ),
  },
  (input) => client.assetMetadata(input.asset_ref, input.session_id, input.version),
);

if (surface === "local") {
  registerJsonTool(
    "asset.fetch",
    "Download one exact binary workspace entry into the MCP adapter's private asset root. " +
    "The tool verifies the streamed size and SHA-256 and returns only a local path plus integrity metadata; " +
    "asset bytes and base64 are never returned to model context.",
    {
      session_id: reference.describe(
        "Exact session:... reference returned by memory.open and retained for workspace continuity.",
      ),
      asset_ref: assetReference,
      version: z.number().int().positive().optional().describe(
        "Optional exact historical version. Metadata and bytes are both fetched at this version.",
      ),
    },
    (input) => client.fetchAsset(input.asset_ref, input.session_id, input.version),
  );
}

registerJsonTool(
  "briefing.publish",
  "Publish or revise one typed briefing edition; Brunn renders the canonical Markdown entry " +
  "and updates the delivered-story ledger. Republishing the same date and edition revises the " +
  "same entry.",
  {
    date: editionDate,
    edition: z.string().regex(/^[a-z0-9][a-z0-9-]{1,31}$/).describe(
      "Lowercase edition slug such as morning.",
    ),
    timezone: z.string().max(64).optional().describe(
      "IANA timezone name used to render generated-at times. Omit this field for the service default.",
    ),
    generated_at: z.string().max(64).optional().describe(
      "Exact RFC3339 timestamp when the briefing content was generated. Omit this field to use the publish time.",
    ),
    summary_md: z.array(z.string().max(1_000)).max(12).optional().describe(
      "30-second version: one Markdown bullet per line, most important first.",
    ),
    sections: z.array(briefingSection).max(24).optional(),
    omitted: z.array(briefingOmission).max(64).optional().describe(
      "Stories researched but deliberately not delivered, with the reason; recorded as suppressions in the ledger.",
    ),
    idempotency_key: z.string().min(1).max(240).optional(),
    expected_version: z.number().int().nonnegative().optional().describe(
      "Supply expected_version only when preventing a known stale overwrite matters.",
    ),
  },
  (input) => client.request("/v1/workspace/briefings/publish", input),
);

registerJsonTool(
  "briefing.dedupe",
  "Check candidate stories against the delivered-story ledger before publishing; returns exact " +
  "URL and story-key matches with delivery history, near matches, and a verdict hint per candidate.",
  {
    candidates: z.array(dedupeCandidate).min(1).max(64),
  },
  (input) => client.request("/v1/workspace/briefings/dedupe-check", input),
);

registerJsonTool(
  "briefing.topics",
  "Read the parsed briefing topics snapshot plus pending go-deeper requests and the recent feedback tail.",
  {
    session_id: reference.describe(
      "Exact session:... reference returned by memory.open.",
    ),
  },
  () => client.request("/v1/workspace/briefings/topics"),
);

registerJsonTool(
  "notification.publish",
  "Publish one durable user alert for the authenticated owner. Brunn deduplicates by " +
  "event_key, records the private inbox detail, and independently queues eligible device deliveries.",
  {
    event_key: z.string().min(1).max(200).describe(
      "Stable semantic identity shared by Codex and Aether. Reuse it only for the same alert content.",
    ),
    correlation_id: z.string().min(1).max(200).describe(
      "Stable correlation identity for the producing run, briefing, incident, or decision chain.",
    ),
    kind: z.enum(["briefing_ready", "news_alert", "correction", "operational"]),
    importance: z.enum(["normal", "important"]),
    title: z.string().min(1).max(240),
    body: z.string().min(1).max(20_000),
    source: notificationSource.optional().describe(
      "Exact durable source and optional pinned version supporting this attention decision.",
    ),
    target: notificationTarget.describe(
      "Typed in-app destination. Use briefing or entry when an exact durable target exists.",
    ),
    occurred_at: z.string().min(1).max(64).optional(),
    expires_at: z.string().min(1).max(64).optional().describe(
      "Optional RFC3339 expiry, no more than seven days after occurred_at. Omit for the 24-hour default.",
    ),
  },
  (input) => client.request("/v1/workspace/notifications/publish", input),
);

registerJsonTool(
  "task.capture",
  "Capture one or many owner tasks from raw text without exposing the backlog. Before inferring, consult "
  + "task.corrections, task.contexts, and the project registry from project.list. Infer project aliases; "
  + "call means phone; buy, pick up, or drop off means errands; needs Nyx means home; renewal, expiry, "
  + "charges, or lost value may make a date hard; and infer evidenced cost or an obvious estimate. Source "
  + "every enrichment, never overwrite an owner value, and preserve the original sentence. Ask at most one "
  + "clarifying question, only for consequential hard/soft ambiguity. This tool never returns the backlog.",
  {
    idempotency_key: taskIdempotencyKey,
    items: z.array(taskCaptureItem).min(1).max(25),
  },
  (input) => client.request("POST", "/v1/workspace/tasks/capture", input),
);

registerJsonTool(
  "task.candidates",
  "Return deterministic ranked tasks with reasons and provenance markers using context AND semantics. "
  + "Defaults to the bounded next five; next accepts at most 25. Urgent returns every visible tier-1/2 "
  + "task and must not be given a limit. Triage is bounded to ten. Use all only for an explicit owner request, "
  + "with deliberate_all=true and cursor pagination. The status, context, date_type, and source filters are "
  + "valid only with view=all. as_of exists for deterministic testing; otherwise omit it.",
  {
    view: z.enum(["next", "urgent", "triage", "all"]).default("next"),
    limit: z.number().int().min(1).max(25).optional().describe(
      "Omit for service defaults (next five, triage ten). Omit for urgent so every tier-1/2 item returns.",
    ),
    contexts_available: contextList.optional().describe(
      "Every required context must be present. Omit or pass [] when no context is available.",
    ),
    project: taskSlug.optional(),
    context: contextSlug.optional().describe(
      "Exact required-context slug filter. Valid only with view=all.",
    ),
    status: z.enum(["all", "open", "waiting", "done", "dropped"]).optional().describe(
      "Exact task status filter, or all. Valid only with view=all.",
    ),
    date_type: z.enum(["all", "hard", "cost", "soft", "none"]).optional().describe(
      "Filter by projected hard, cost, soft, or no date/cost signal, or all. Valid only with view=all.",
    ),
    source: z.enum(["all", "owner", "agent", "derived", "todoist"]).optional().describe(
      "Filter by any matching provenance source family, or all. Valid only with view=all.",
    ),
    include_waiting: z.boolean().default(false),
    include_parked: z.boolean().default(false),
    as_of: rfc3339Timestamp.optional().describe(
      "Explicit evaluation instant for deterministic testing. Omit in normal use.",
    ),
    cursor: taskRef.optional(),
    deliberate_all: z.literal(true).optional().describe(
      "Required only for view=all, after the owner explicitly asks to see the full paginated list.",
    ),
  },
  (input) => {
    if (input.view === "all" && input.deliberate_all !== true) {
      throw new Error("view=all requires deliberate_all=true after an explicit owner request");
    }
    if (input.view !== "all" && input.deliberate_all !== undefined) {
      throw new Error("deliberate_all is valid only with view=all");
    }
    if (
      input.view !== "all"
      && [input.status, input.context, input.date_type, input.source].some(
        (value) => value !== undefined,
      )
    ) {
      throw new Error("status, context, date_type, and source are valid only with view=all");
    }
    if (input.view === "urgent" && input.limit !== undefined) {
      throw new Error("urgent is unbounded across visible tier-1/2 tasks; omit limit");
    }
    if (input.view === "triage" && input.limit !== undefined && input.limit > 10) {
      throw new Error("triage is bounded to at most ten tasks");
    }
    const query = new URLSearchParams();
    query.set("view", input.view);
    appendQuery(query, "limit", input.limit);
    for (const context of input.contexts_available ?? []) {
      query.append("contexts_available", context);
    }
    appendQuery(query, "project", input.project);
    appendQuery(query, "context", input.context);
    appendQuery(query, "status", input.status);
    appendQuery(query, "date_type", input.date_type);
    appendQuery(query, "source", input.source);
    query.set("include_waiting", String(input.include_waiting));
    query.set("include_parked", String(input.include_parked));
    appendQuery(query, "as_of", input.as_of);
    appendQuery(query, "cursor", input.cursor);
    appendQuery(query, "deliberate_all", input.deliberate_all);
    return client.request("GET", withQuery("/v1/workspace/tasks/candidates", query));
  },
);

registerJsonTool(
  "task.update",
  "Apply exactly one sourced correction or action to one task with optimistic concurrency. Actions are "
  + "complete, reopen, snooze, drop, wait_on, unpark, pin_today, unpin, confirm_hard, and "
  + "downgrade_to_soft. Every operation requires source; complete also requires completed_via and returns "
  + "done_today_count. Replay an ambiguous result with the identical idempotency_key and payload.",
  {
    task_ref: taskRef,
    expected_version: z.number().int().positive(),
    idempotency_key: taskIdempotencyKey,
    operation: taskUpdateOperation,
  },
  (input) => {
    const { task_ref, ...body } = input;
    return client.request(
      "PATCH",
      `/v1/workspace/tasks/${encodeURIComponent(task_ref)}`,
      body,
    );
  },
);

registerJsonTool(
  "task.corrections",
  "Read a bounded, recent corrections log for enrichment feedback. Consult this before task.capture; "
  + "Brunn records corrections but never turns them into hidden learned logic.",
  {
    task_ref: taskRef.optional().describe(
      "Optional exact task_ref filter when reviewing corrections for one task.",
    ),
    limit: z.number().int().min(1).max(100).default(20),
    cursor: printableUtf8String(4_096).optional(),
  },
  (input) => {
    const query = new URLSearchParams();
    appendQuery(query, "task_ref", input.task_ref);
    query.set("limit", String(input.limit));
    appendQuery(query, "cursor", input.cursor);
    return client.request("GET", withQuery("/v1/workspace/tasks/corrections", query));
  },
);

registerJsonTool(
  "task.contexts",
  "List, create, explicitly merge, archive, or set available task contexts. Create performs exact alias, "
  + "shared-token, and small-edit checks. A near match is a successful status=needs_review response with "
  + "suggested_existing and no write; ask the owner one question, then retry with confirm_new=true only if "
  + "they want a distinct context. Never merge automatically; merge is explicit and audited. List first to "
  + "obtain expected_from_version and expected_into_version for merge, expected_version for archive, and "
  + "the surface-default expected_version; zero creates an unseeded surface and must not overwrite one.",
  {
    operation: contextOperation,
  },
  (input) => {
    const operation = input.operation;
    if (operation.type === "list") {
      const query = new URLSearchParams({
        include_archived: String(operation.include_archived),
        limit: String(operation.limit),
      });
      appendQuery(query, "cursor", operation.cursor);
      return client.request("GET", withQuery("/v1/workspace/contexts", query));
    }
    if (operation.type === "create") {
      const { type: _type, ...body } = operation;
      return client.request("POST", "/v1/workspace/contexts", body);
    }
    if (operation.type === "merge") {
      const { type: _type, ...body } = operation;
      return client.request("POST", "/v1/workspace/contexts/merge", body);
    }
    if (operation.type === "archive") {
      const { type: _type, slug, ...body } = operation;
      return client.request(
        "PATCH",
        `/v1/workspace/contexts/${encodeURIComponent(slug)}`,
        body,
      );
    }
    const { type: _type, surface, ...body } = operation;
    return client.request(
      "PUT",
      `/v1/workspace/contexts/available/${encodeURIComponent(surface)}`,
      body,
    );
  },
);

registerJsonTool(
  "task.done_summary",
  "Return bounded completed tasks for an explicit inclusive owner-local date range. Supply both from and "
  + "through, or neither for owner-local Done today. Use an explicit range for weekly summaries; there is no "
  + "ambiguous implicit week. as_of exists only for deterministic testing.",
  {
    from: localDate.optional(),
    through: localDate.optional(),
    as_of: rfc3339Timestamp.optional(),
    limit: z.number().int().min(1).max(100).default(25),
    cursor: printableUtf8String(4_096).optional(),
  },
  (input) => {
    if ((input.from === undefined) !== (input.through === undefined)) {
      throw new Error("done summary requires both from and through, or neither for Done today");
    }
    if (input.from !== undefined && input.through !== undefined && input.from > input.through) {
      throw new Error("done summary from date must not be after through date");
    }
    const query = new URLSearchParams();
    appendQuery(query, "from", input.from);
    appendQuery(query, "through", input.through);
    appendQuery(query, "as_of", input.as_of);
    query.set("limit", String(input.limit));
    appendQuery(query, "cursor", input.cursor);
    return client.request("GET", withQuery("/v1/workspace/tasks/done-summary", query));
  },
);

registerJsonTool(
  "task.settings",
  "Get or optimistically update deterministic task windows, timezone, guard leads, quiet-hours override, "
  + "and per-surface context defaults. The mixed tool is conservatively annotated as a mutation; update "
  + "requires a durable idempotency key and expected version.",
  {
    operation: taskSettingsOperation,
  },
  (input) => {
    if (input.operation.type === "get") {
      return client.request("GET", "/v1/workspace/tasks/settings");
    }
    const { type: _type, ...body } = input.operation;
    return client.request("PUT", "/v1/workspace/tasks/settings", body);
  },
);

registerJsonTool(
  "project.register",
  "Create or update one open-vocabulary project registry record. Register aliases and optional hub_path "
  + "or repo_path so checkpoint linkage can use deterministic longest-prefix fallback. This stores registry "
  + "metadata, not a task list.",
  {
    slug: taskSlug,
    title: z.string().min(1).max(200),
    aliases: z.array(z.string().min(1).max(160)).max(32).optional(),
    description: z.string().min(1).max(2_000).optional(),
    hub_path: z.string().min(1).max(4_096).refine(
      (value) => !value.startsWith("/") && !value.split("/").includes(".."),
      "hub_path must be a safe workspace-relative path",
    ).optional(),
    repo_path: z.string().min(1).max(4_096).optional(),
    archived: z.boolean().optional(),
    source: taskWriteSource,
    expected_version: z.number().int().nonnegative().optional(),
    idempotency_key: taskIdempotencyKey,
  },
  (input) => {
    const { slug, ...body } = input;
    return client.request(
      "PUT",
      `/v1/workspace/projects/${encodeURIComponent(slug)}`,
      body,
    );
  },
);

registerJsonTool(
  "project.list",
  "List the bounded project registry with deterministic current interest and activity. It never returns "
  + "a wall of tasks; use project.state for one project's checkpoint and rollups.",
  {
    include_archived: z.boolean().default(false),
    limit: z.number().int().min(1).max(100).default(50),
    cursor: taskSlug.optional(),
    as_of: rfc3339Timestamp.optional(),
  },
  (input) => {
    const query = new URLSearchParams({
      include_archived: String(input.include_archived),
      limit: String(input.limit),
    });
    appendQuery(query, "cursor", input.cursor);
    appendQuery(query, "as_of", input.as_of);
    return client.request("GET", withQuery("/v1/workspace/projects", query));
  },
);

registerJsonTool(
  "project.state",
  "Return one project's latest linked checkpoint objective and current state, next actions, open questions, "
  + "checkpoint time, next three candidates, urgent and parked counts, waiting items with ages, current "
  + "interest, and last activity. It never returns the full task backlog.",
  {
    slug: taskSlug,
    as_of: rfc3339Timestamp.optional(),
  },
  (input) => {
    const query = new URLSearchParams();
    appendQuery(query, "as_of", input.as_of);
    return client.request(
      "GET",
      withQuery(`/v1/workspace/projects/${encodeURIComponent(input.slug)}/state`, query),
    );
  },
);

registerJsonTool(
  "project.set_interest",
  "Set an optimistic, sourced hot, normal, or parked project-interest override. The explicit override lasts "
  + "14 days, then deterministic activity-derived interest resumes.",
  {
    slug: taskSlug,
    interest: z.enum(["hot", "normal", "parked"]),
    source: taskWriteSource,
    expected_version: z.number().int().positive(),
    idempotency_key: taskIdempotencyKey,
  },
  (input) => {
    const { slug, ...body } = input;
    return client.request(
      "PUT",
      `/v1/workspace/projects/${encodeURIComponent(slug)}/interest`,
      body,
    );
  },
);

registerJsonTool(
  "task.sync_status",
  "Read content-free Todoist pull status: environment gate, saved and effective mode, token-configured "
  + "boolean, last run/outcome/error summary, and next run. It never returns a token or task content.",
  {},
  () => client.request("GET", "/v1/workspace/integrations/todoist/status"),
);

registerJsonTool(
  "secret.put",
  "Store or replace one named secret for the authenticated owner in the encrypted vault. "
  + "Use this when the user hands over a credential, API key, or token that agents will need "
  + "again; replacing an existing name rotates it to a new version. Never store secrets in "
  + "memory files or captures.",
  {
    name: secretName,
    value: z.string().min(1).max(64 * 1024).describe(
      "The secret value exactly as provided, including any newlines. Text, JSON, and "
      + "multiline keys up to 64 KiB are accepted.",
    ),
    description: z.string().min(1).max(1_000).optional().describe(
      "Non-secret usage note, e.g. what the credential unlocks. Omitting it on replace "
      + "keeps the existing note.",
    ),
  },
  (input) => client.request("/v1/workspace/secrets/put", input),
);

registerJsonTool(
  "secret.get",
  "Retrieve one named secret's plaintext value from the encrypted vault. Treat the returned "
  + "value as sensitive: use it for the requested action, and never write it into memory "
  + "files, captures, checkpoints, or other durable output.",
  {
    name: secretName,
  },
  (input) => client.request("/v1/workspace/secrets/get", input),
);

registerJsonTool(
  "secret.list",
  "List stored secret names and metadata (description, version, timestamps, last use). "
  + "Values are never included. Use this to discover whether a needed credential already "
  + "exists before asking the user for it.",
  {},
  () => client.request("/v1/workspace/secrets"),
);

registerJsonTool(
  "secret.delete",
  "Permanently delete one named secret from the encrypted vault. The encrypted value is "
  + "removed immediately; content-free access history is retained.",
  {
    name: secretName,
  },
  (input) => client.request("/v1/workspace/secrets/delete", input),
);

  if (messagingEnabled) {
    registerMessagingTools(server, client, { includeStructuredContent });
  }

  return server;
}

function registerJsonToolOnServer<Shape extends z.ZodRawShape>(
  server: McpServer,
  includeStructuredContent: boolean,
  name: string,
  description: string,
  inputSchema: Shape,
  invoke: (input: z.infer<z.ZodObject<Shape>>) => Promise<ApiResponse>,
): void {
  const callback = async (input: z.infer<z.ZodObject<Shape>>) => {
    try {
      const response = await invoke(input);
      const body = compactReasoningResponse(name, response.body);
      await traceOperation(name, response.status, response.elapsedMs, response.body, body);
      return {
        content: [{ type: "text" as const, text: JSON.stringify(body) }],
        ...(includeStructuredContent ? { structuredContent: body } : {}),
      };
    } catch (error) {
      const body = error instanceof BrunnApiError
        ? error.body
        : { error: { code: "adapter_error", message: errorMessage(error) } };
      await traceOperation(
        name,
        error instanceof BrunnApiError ? error.status : 0,
        0,
        body,
        body,
      );
      return {
        isError: true,
        content: [{ type: "text" as const, text: JSON.stringify(body) }],
        ...(includeStructuredContent ? { structuredContent: body } : {}),
      };
    }
  };
  // McpServer validates the raw shape before calling this function. Its generic
  // callback type does not preserve a reusable helper's Zod shape inference.
  const readOnly = !new Set([
    "asset.upload_url",
    "document.publish",
    "memory.capture",
    "memory.write",
    "memory.checkpoint",
    "memory.stage",
    "briefing.publish",
    "notification.publish",
    "task.capture",
    "task.update",
    "task.contexts",
    "task.settings",
    "project.register",
    "project.set_interest",
    "location.rederive",
    "secret.put",
    "secret.delete",
  ]).has(name);
  const idempotent = readOnly
    || name === "memory.checkpoint"
    || name === "notification.publish"
    || name === "task.capture"
    || name === "task.update"
    || name === "task.contexts"
    || name === "task.settings"
    || name === "project.register"
    || name === "project.set_interest"
    || name === "location.rederive";
  server.registerTool(name, {
    description,
    inputSchema,
    annotations: {
      readOnlyHint: readOnly,
      destructiveHint: false,
      idempotentHint: idempotent,
      openWorldHint: false,
    },
  }, callback as never);
}

async function runStdioServer(): Promise<void> {
  const client = new BrunnApiClient(
    process.env.BRUNN_API_URL ?? "http://api:18110",
    requiredEnvironment("BRUNN_API_TOKEN"),
    fetch,
    evaluationHeaders(),
  );
  await createBrunnMcpServer(client).connect(new StdioServerTransport());
}

if (
  process.argv[1] !== undefined
  && import.meta.url === pathToFileURL(process.argv[1]).href
) {
  await runStdioServer();
}

function requiredEnvironment(name: string): string {
  const value = process.env[name];
  if (!value) {
    process.stderr.write(`${name} is required\n`);
    process.exit(78);
  }
  return value;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function appendQuery(
  query: URLSearchParams,
  name: string,
  value: string | number | boolean | undefined,
): void {
  if (value !== undefined) {
    query.set(name, String(value));
  }
}

function withQuery(path: string, query: URLSearchParams): string {
  const serialized = query.toString();
  return serialized.length === 0 ? path : `${path}?${serialized}`;
}

function printableUtf8String(maxBytes: number) {
  return z.string().min(1).max(maxBytes).refine(
    (value) => Buffer.byteLength(value, "utf8") <= maxBytes
      && !/[\u0000-\u001f\u007f-\u009f]/u.test(value),
    `value must contain at most ${maxBytes} UTF-8 bytes and no control characters`,
  );
}

function serializedUtf8Length(value: unknown): number {
  try {
    return Buffer.byteLength(JSON.stringify(value), "utf8");
  } catch {
    return Number.POSITIVE_INFINITY;
  }
}

function evaluationHeaders(): Record<string, string> {
  const headers: Record<string, string> = {};
  if (process.env.BRUNN_EVAL_RUN) {
    headers["x-brunn-eval-run"] = process.env.BRUNN_EVAL_RUN;
  }
  if (process.env.BRUNN_EVAL_CASE) {
    headers["x-brunn-eval-case"] = process.env.BRUNN_EVAL_CASE;
  }
  return headers;
}

async function traceOperation(
  operation: string,
  httpStatus: number,
  elapsedMs: number,
  response: Record<string, unknown>,
  rendered: Record<string, unknown>,
): Promise<void> {
  const tracePath = process.env.BRUNN_MCP_TRACE_PATH;
  if (!tracePath) {
    return;
  }
  const renderedText = JSON.stringify(rendered);
  const sourceTextChars = countSourceTextChars(rendered);
  const binaryBytes = operation === "asset.fetch"
    ? Number(findField(rendered, ["size_bytes"]) ?? 0)
    : 0;
  const sourcePaths = collectSourcePaths(rendered);
  const record = {
    at: new Date().toISOString(),
    operation,
    http_status: httpStatus,
    elapsed_ms: Math.round(elapsedMs * 1_000) / 1_000,
    result_chars: renderedText.length,
    source_text_chars: sourceTextChars,
    metadata_chars: Math.max(0, renderedText.length - sourceTextChars),
    request_id: findField(response, ["request_id"]),
    service_status: findField(response, ["status"]),
    session_id: findField(response, ["session_id"]),
    corpus_revision: findField(response, ["corpus_revision", "revision_id"]),
    checkpoint_id: findField(response, ["checkpoint_id"]),
    http_calls: operation === "asset.fetch" ? 2 : 1,
    binary_bytes: Number.isSafeInteger(binaryBytes) && binaryBytes >= 0
      ? binaryBytes
      : 0,
    source_paths: sourcePaths,
    asset_ref: (operation === "asset.fetch" || operation === "asset.metadata")
      ? findField(rendered, ["asset_ref"])
      : undefined,
    asset_version: (operation === "asset.fetch" || operation === "asset.metadata")
      ? findField(rendered, ["version"])
      : undefined,
    asset_content_hash: (operation === "asset.fetch" || operation === "asset.metadata")
      ? findField(rendered, ["content_hash"])
      : undefined,
    asset_size_bytes: (operation === "asset.fetch" || operation === "asset.metadata")
      ? findField(rendered, ["size_bytes"])
      : undefined,
    asset_local_path: operation === "asset.fetch"
      ? findField(rendered, ["local_path"])
      : undefined,
  };
  try {
    await appendFile(tracePath, `${JSON.stringify(record)}\n`, { encoding: "utf8", mode: 0o600 });
  } catch {
    // Evaluation telemetry must never alter the result of a memory operation.
  }
}

function findField(value: unknown, names: string[]): unknown {
  if (Array.isArray(value)) {
    for (const child of value) {
      const match = findField(child, names);
      if (match !== undefined && match !== null) {
        return match;
      }
    }
    return undefined;
  }
  if (typeof value !== "object" || value === null) {
    return undefined;
  }
  const record = value as Record<string, unknown>;
  for (const name of names) {
    if (record[name] !== undefined && record[name] !== null) {
      return record[name];
    }
  }
  for (const child of Object.values(record)) {
    const match = findField(child, names);
    if (match !== undefined && match !== null) {
      return match;
    }
  }
  return undefined;
}

function countSourceTextChars(value: unknown): number {
  if (Array.isArray(value)) {
    return value.reduce((total, child) => total + countSourceTextChars(child), 0);
  }
  if (typeof value !== "object" || value === null) {
    return 0;
  }
  let total = 0;
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    if ((key === "content" || key === "text") && typeof child === "string") {
      total += child.length;
    } else {
      total += countSourceTextChars(child);
    }
  }
  return total;
}

function collectSourcePaths(value: unknown): string[] {
  const paths = new Set<string>();
  const visit = (child: unknown): void => {
    if (Array.isArray(child)) {
      child.forEach(visit);
      return;
    }
    if (typeof child !== "object" || child === null) {
      return;
    }
    for (const [key, nested] of Object.entries(child as Record<string, unknown>)) {
      if (
        key === "path"
        && typeof nested === "string"
        && nested.length > 0
        && !nested.startsWith("/")
      ) {
        paths.add(nested.replace(/^\.\//, ""));
      }
      visit(nested);
    }
  };
  visit(value);
  return [...paths].sort();
}
