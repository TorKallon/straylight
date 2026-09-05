import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

test("stdio server negotiates and exposes the complete typed memory surface", async (context) => {
  const environment = Object.fromEntries(
    Object.entries(process.env).filter((entry): entry is [string, string] => entry[1] !== undefined),
  );
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [fileURLToPath(new URL("./index.js", import.meta.url))],
    env: {
      ...environment,
      BRUNN_API_TOKEN: "protocol-test-token",
      BRUNN_API_URL: "http://127.0.0.1:1",
      BRUNN_MCP_RETRY_BACKOFF_MS: "1,1,1,1,1,1",
    },
  });
  const client = new Client({ name: "brunn-adapter-test", version: "0.1.0" });
  context.after(async () => {
    await client.close();
  });

  await client.connect(transport);
  const response = await client.listTools();

  const names = response.tools.map((tool) => tool.name).sort();
  const expectedNames = [
    "asset.fetch",
    "asset.list",
    "asset.metadata",
    "asset.upload_url",
    "briefing.dedupe",
    "briefing.publish",
    "briefing.topics",
    "document.get",
    "document.publish",
    "location.presence",
    "location.rederive",
    "memory.capture",
    "memory.changes",
    "memory.checkpoint",
    "memory.open",
    "memory.query",
    "memory.read",
    "memory.stage",
    "memory.status",
    "memory.write",
    "notification.publish",
    "project.list",
    "project.register",
    "project.set_interest",
    "project.state",
    "secret.delete",
    "secret.get",
    "secret.list",
    "secret.put",
    "task.candidates",
    "task.capture",
    "task.contexts",
    "task.corrections",
    "task.done_summary",
    "task.settings",
    "task.sync_status",
    "task.update",
  ];
  if (process.env.BRUNN_MESSAGING_ENABLED === "true") {
    expectedNames.push(
      "agent.list",
      "message.list",
      "message.read",
      "message.send",
      "message.wait",
    );
  }
  for (const name of expectedNames) assert.ok(names.includes(name), `missing tool ${name}`);
  assert.equal(response.tools.every((tool) => tool.inputSchema.type === "object"), true);
  const open = response.tools.find((tool) => tool.name === "memory.open");
  assert.ok(open);
  const resumeCheckpointRef = open.inputSchema.properties?.resume_checkpoint_ref as {
    description?: string;
  } | undefined;
  assert.match(resumeCheckpointRef?.description ?? "", /Omit this field/);
  assert.match(resumeCheckpointRef?.description ?? "", /never invent/);
  const openModes = open.inputSchema.properties?.modes as {
    description?: string;
  } | undefined;
  assert.match(openModes?.description ?? "", /exact and lexical/);
  const query = response.tools.find((tool) => tool.name === "memory.query");
  assert.ok(query);
  assert.match(query.description ?? "", /current workspace files/);
  const queries = query.inputSchema.properties?.queries as {
    maxItems?: number;
    items?: {
      properties?: {
        limit?: { default?: number };
        modes?: { description?: string };
      };
    };
  } | undefined;
  assert.equal(queries?.maxItems, 16);
  assert.equal(queries?.items?.properties?.limit?.default, 8);
  assert.match(queries?.items?.properties?.modes?.description ?? "", /hybrid search/);
  const queryTokenBudget = query.inputSchema.properties?.token_budget as {
    description?: string;
  } | undefined;
  assert.match(queryTokenBudget?.description ?? "", /response budget/);
  const read = response.tools.find((tool) => tool.name === "memory.read");
  assert.ok(read);
  assert.match(read.description ?? "", /exact reads/);
  const requests = read.inputSchema.properties?.requests as {
    items?: {
      properties?: {
        ref?: { description?: string };
        path?: { description?: string };
      };
    };
  } | undefined;
  assert.equal("before" in (requests?.items?.properties ?? {}), false);
  assert.equal("after" in (requests?.items?.properties ?? {}), false);
  assert.match(requests?.items?.properties?.ref?.description ?? "", /verbatim/);
  assert.match(requests?.items?.properties?.path?.description ?? "", /Never synthesize/);
  const checkpoint = response.tools.find((tool) => tool.name === "memory.checkpoint");
  assert.ok(checkpoint);
  assert.equal(checkpoint.annotations?.idempotentHint, true);
  assert.equal(
    checkpoint.inputSchema.required?.includes("idempotency_key"),
    true,
    "MCP checkpoints must opt into explicit durable cross-session replay",
  );
  const checkpointIdempotencyKey = checkpoint.inputSchema.properties?.idempotency_key as {
    maxLength?: number;
  } | undefined;
  assert.equal(checkpointIdempotencyKey?.maxLength, 256);
  assert.equal(
    (checkpoint.inputSchema.properties?.session_id as { maxLength?: number } | undefined)
      ?.maxLength,
    256,
  );
  assert.equal(
    (checkpoint.inputSchema.properties?.parent_checkpoint_id as {
      maxLength?: number;
    } | undefined)?.maxLength,
    256,
  );
  const sourceRefs = checkpoint.inputSchema.properties?.source_refs as {
    maxItems?: number;
    items?: { description?: string; maxLength?: number };
  } | undefined;
  assert.equal(sourceRefs?.maxItems, 64);
  assert.equal(sourceRefs?.items?.maxLength, 4_096);
  assert.match(sourceRefs?.items?.description ?? "", /Markdown path/);
  const checkpointState = checkpoint.inputSchema.properties?.state as {
    properties?: {
      objective?: { maxLength?: number };
      project?: { maxLength?: number; pattern?: string };
      decisions?: { maxItems?: number; items?: { maxLength?: number } };
      artifacts?: { maxItems?: number; items?: { maxLength?: number } };
      state_refs?: { maxItems?: number; items?: { maxLength?: number } };
    };
  } | undefined;
  assert.equal(checkpointState?.properties?.objective?.maxLength, 4 * 1024 * 1024);
  assert.equal(checkpointState?.properties?.project?.maxLength, 100);
  assert.match(checkpointState?.properties?.project?.pattern ?? "", /a-z0-9/);
  assert.equal(checkpointState?.properties?.decisions?.maxItems, 4_096);
  assert.equal(
    checkpointState?.properties?.decisions?.items?.maxLength,
    4 * 1024 * 1024,
  );
  assert.equal(checkpointState?.properties?.artifacts?.maxItems, 4_096);
  assert.equal(
    checkpointState?.properties?.artifacts?.items?.maxLength,
    4 * 1024 * 1024,
  );
  assert.equal(checkpointState?.properties?.state_refs?.maxItems, 4_096);
  assert.equal(checkpointState?.properties?.state_refs?.items?.maxLength, 4_096);
  const changes = response.tools.find((tool) => tool.name === "memory.changes");
  assert.ok(changes);
  assert.equal(
    (changes.inputSchema.properties?.since_generation as { default?: number } | undefined)
      ?.default,
    0,
  );
  assert.equal(
    (changes.inputSchema.properties?.limit as { default?: number } | undefined)
      ?.default,
    200,
  );
  const write = response.tools.find((tool) => tool.name === "memory.write");
  assert.ok(write);
  assert.deepEqual(
    [...(write.inputSchema.required ?? [])].sort(),
    ["content", "path"],
  );
  const assetList = response.tools.find((tool) => tool.name === "asset.list");
  assert.ok(assetList);
  assert.deepEqual(
    [...(assetList.inputSchema.required ?? [])].sort(),
    ["session_id"],
  );
  assert.equal(
    (assetList.inputSchema.properties?.offset as { default?: number } | undefined)
      ?.default,
    0,
  );
  assert.equal(
    (assetList.inputSchema.properties?.limit as { default?: number } | undefined)
      ?.default,
    100,
  );
  const assetMetadata = response.tools.find((tool) => tool.name === "asset.metadata");
  assert.ok(assetMetadata);
  assert.deepEqual(
    [...(assetMetadata.inputSchema.required ?? [])].sort(),
    ["asset_ref", "session_id"],
  );
  assert.ok(assetMetadata.inputSchema.properties?.version);
  const assetFetch = response.tools.find((tool) => tool.name === "asset.fetch");
  assert.ok(assetFetch);
  assert.match(assetFetch.description ?? "", /bytes and base64 are never returned/);
  assert.deepEqual(
    [...(assetFetch.inputSchema.required ?? [])].sort(),
    ["asset_ref", "session_id"],
  );
  const briefingPublish = response.tools.find((tool) => tool.name === "briefing.publish");
  assert.ok(briefingPublish);
  assert.deepEqual(
    [...(briefingPublish.inputSchema.required ?? [])].sort(),
    ["date", "edition"],
  );
  const publishDate = briefingPublish.inputSchema.properties?.date as {
    description?: string;
  } | undefined;
  assert.match(publishDate?.description ?? "", /YYYY-MM-DD/);
  const publishSections = briefingPublish.inputSchema.properties?.sections as {
    maxItems?: number;
    items?: {
      properties?: {
        items?: {
          maxItems?: number;
          items?: {
            properties?: {
              story?: { properties?: { key?: { description?: string } } };
            };
          };
        };
      };
    };
  } | undefined;
  assert.equal(publishSections?.maxItems, 24);
  assert.equal(publishSections?.items?.properties?.items?.maxItems, 32);
  assert.match(
    publishSections?.items?.properties?.items?.items?.properties?.story?.properties?.key
      ?.description ?? "",
    /never invent/,
  );
  const briefingDedupe = response.tools.find((tool) => tool.name === "briefing.dedupe");
  assert.ok(briefingDedupe);
  const dedupeCandidates = briefingDedupe.inputSchema.properties?.candidates as {
    minItems?: number;
    maxItems?: number;
    items?: {
      properties?: {
        urls?: { maxItems?: number };
        story_key?: { description?: string };
      };
    };
  } | undefined;
  assert.equal(dedupeCandidates?.minItems, 1);
  assert.equal(dedupeCandidates?.maxItems, 64);
  assert.equal(dedupeCandidates?.items?.properties?.urls?.maxItems, 8);
  assert.match(
    dedupeCandidates?.items?.properties?.story_key?.description ?? "",
    /verbatim/,
  );
  const briefingTopics = response.tools.find((tool) => tool.name === "briefing.topics");
  assert.ok(briefingTopics);
  assert.deepEqual(
    [...(briefingTopics.inputSchema.required ?? [])].sort(),
    ["session_id"],
  );
  const notificationPublish = response.tools.find(
    (tool) => tool.name === "notification.publish",
  );
  assert.ok(notificationPublish);
  assert.deepEqual(
    [...(notificationPublish.inputSchema.required ?? [])].sort(),
    ["body", "correlation_id", "event_key", "importance", "kind", "target", "title"],
  );
  const notificationTarget = notificationPublish.inputSchema.properties?.target as {
    description?: string;
    oneOf?: Array<{
      properties?: Record<string, { const?: string; maxLength?: number }>;
    }>;
  } | undefined;
  assert.match(notificationTarget?.description ?? "", /Typed in-app destination/);
  const notificationProperties = notificationPublish.inputSchema.properties as Record<
    string,
    { maxLength?: number; properties?: Record<string, { maxLength?: number }> }
  >;
  assert.equal(notificationProperties.event_key?.maxLength, 200);
  assert.equal(notificationProperties.correlation_id?.maxLength, 200);
  assert.equal(notificationProperties.title?.maxLength, 240);
  assert.equal(notificationProperties.body?.maxLength, 20_000);
  assert.equal(notificationProperties.source?.properties?.type?.maxLength, 64);
  assert.equal(notificationProperties.source?.properties?.ref?.maxLength, 500);
  assert.equal(notificationProperties.source?.properties?.version_ref?.maxLength, 500);
  const targetVariants = notificationTarget?.oneOf ?? [];
  const briefingTarget = targetVariants.find(
    (variant) => variant.properties?.type?.const === "briefing",
  );
  const entryTarget = targetVariants.find(
    (variant) => variant.properties?.type?.const === "entry",
  );
  assert.equal(briefingTarget?.properties?.edition?.maxLength, 64);
  assert.equal(briefingTarget?.properties?.item_id?.maxLength, 200);
  assert.equal(entryTarget?.properties?.entry_ref?.maxLength, 500);

  const call = await client.callTool({ name: "memory.status", arguments: {} });
  assert.equal(call.isError, true);
  assert.equal(call.structuredContent, undefined);
  assert.equal(Array.isArray(call.content), true);
  const text = (call.content as Array<{ type: string; text?: string }>)[0];
  assert.equal(text?.type, "text");
  if (text?.type === "text" && text.text) {
    const failure = JSON.parse(text.text) as {
      error: { code: string; attempts: number; retryable: boolean };
    };
    assert.equal(failure.error.code, "upstream_unavailable");
    assert.equal(failure.error.attempts, 7);
    assert.equal(failure.error.retryable, true);
  }
});

test("binary MCP tools return metadata or a verified local path, never payload bytes", async () => {
  const assetRef = "entry:019f8530-e5f6-77d3-a373-052ee8cd24bd";
  const sessionId = "session:019f8531-06fa-7fe0-9050-0648d7e8553e";
  const bytes = Buffer.from("literal receipt payload that must stay outside model context");
  const base64 = bytes.toString("base64");
  const digest = createHash("sha256").update(bytes).digest("hex");
  const assetRoot = await mkdtemp(join(tmpdir(), "brunn-state-mcp-protocol-assets-"));
  const requests: string[] = [];
  const httpServer = createServer((request, response) => {
    const requestUrl = new URL(request.url ?? "/", "http://127.0.0.1");
    const requestPath = decodeURIComponent(requestUrl.pathname);
    requests.push(requestUrl.pathname + requestUrl.search);
    if (
      requestPath === `/v1/workspace/binaries/${assetRef}`
    ) {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({
        entry_ref: assetRef,
        version: 2,
        content_hash: `sha256:${digest}`,
        size_bytes: bytes.byteLength,
        media_type: "image/jpeg",
        path: "Trips/receipt.jpg",
      }));
      return;
    }
    if (
      requestPath === `/v1/workspace/binaries/${assetRef}/content`
    ) {
      response.writeHead(200, {
        "content-length": String(bytes.byteLength),
        "content-type": "image/jpeg",
        "x-brunn-state-asset-ref": assetRef,
        "x-brunn-state-asset-version": "2",
        "x-brunn-state-sha256": digest,
      });
      response.write(bytes.subarray(0, 11));
      response.end(bytes.subarray(11));
      return;
    }
    response.writeHead(404, { "content-type": "application/json" });
    response.end(JSON.stringify({ error: { code: "not_found", message: "not found" } }));
  });
  await new Promise<void>((resolve, reject) => {
    httpServer.once("error", reject);
    httpServer.listen(0, "127.0.0.1", () => resolve());
  });
  const address = httpServer.address();
  assert.ok(address && typeof address === "object");

  const environment = Object.fromEntries(
    Object.entries(process.env).filter((entry): entry is [string, string] => entry[1] !== undefined),
  );
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [fileURLToPath(new URL("./index.js", import.meta.url))],
    env: {
      ...environment,
      BRUNN_API_TOKEN: "protocol-test-token",
      BRUNN_API_URL: `http://127.0.0.1:${address.port}`,
      BRUNN_STATE_MCP_ASSET_ROOT: assetRoot,
    },
  });
  const client = new Client({ name: "brunn-state-asset-test", version: "0.1.0" });

  try {
    await client.connect(transport);
    const metadataCall = await client.callTool({
      name: "asset.metadata",
      arguments: { asset_ref: assetRef, session_id: sessionId, version: 2 },
    });
    assert.equal(metadataCall.isError, undefined);
    const metadata = parseToolText(metadataCall.content);
    assert.equal(metadata.entry_ref, assetRef);
    assert.equal(metadata.content_hash, `sha256:${digest}`);

    const fetchCall = await client.callTool({
      name: "asset.fetch",
      arguments: { asset_ref: assetRef, session_id: sessionId, version: 2 },
    });
    assert.equal(fetchCall.isError, undefined);
    assert.equal(fetchCall.structuredContent, undefined);
    const fetched = parseToolText(fetchCall.content);
    assert.deepEqual(Object.keys(fetched).sort(), [
      "content_hash",
      "local_path",
      "media_type",
      "size_bytes",
    ]);
    assert.equal(fetched.content_hash, `sha256:${digest}`);
    assert.equal(fetched.size_bytes, bytes.byteLength);
    assert.equal(fetched.media_type, "image/jpeg");
    assert.deepEqual(await readFile(String(fetched.local_path)), bytes);
    const rendered = JSON.stringify(fetchCall);
    assert.equal(rendered.includes(bytes.toString()), false);
    assert.equal(rendered.includes(base64), false);
    assert.deepEqual(requests, [
      `/v1/workspace/binaries/${encodeURIComponent(assetRef)}?version=2`,
      `/v1/workspace/binaries/${encodeURIComponent(assetRef)}?version=2`,
      `/v1/workspace/binaries/${encodeURIComponent(assetRef)}/content?version=2`,
    ]);
  } finally {
    await client.close().catch(() => undefined);
    await new Promise<void>((resolve) => httpServer.close(() => resolve()));
    await rm(assetRoot, { recursive: true, force: true });
  }
});

function parseToolText(content: unknown): Record<string, unknown> {
  assert.ok(Array.isArray(content));
  const first = content[0] as { type?: string; text?: string } | undefined;
  assert.equal(first?.type, "text");
  if (typeof first?.text !== "string") {
    throw new Error("MCP tool response did not contain text");
  }
  return JSON.parse(first.text) as Record<string, unknown>;
}
