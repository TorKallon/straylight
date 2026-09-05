#!/usr/bin/env node

import assert from "node:assert/strict";
import { createHash, randomBytes } from "node:crypto";
import { readFile } from "node:fs/promises";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

import { expectedRemoteToolNames } from "./messaging-release-profile.mjs";
import { archiveFiles } from "./binary-upload-canary.mjs";

const baseUrl = new URL(process.env.BRUNN_REMOTE_URL ?? "https://brunn.ai/");
const resourceUrl = new URL("/mcp", baseUrl);
const upstreamToken = requiredEnvironment("BRUNN_REMOTE_TOKEN");
const label = requiredEnvironment("BRUNN_REMOTE_LABEL");
const canaryPath = requiredEnvironment("BRUNN_REMOTE_CANARY_PATH");
const marker = requiredEnvironment("BRUNN_REMOTE_MARKER");

const registration = await json(await fetch(new URL("/register", baseUrl), {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({
    redirect_uris: ["http://127.0.0.1/callback"],
    token_endpoint_auth_method: "none",
    grant_types: ["authorization_code", "refresh_token"],
    response_types: ["code"],
    client_name: `${label} production canary`,
  }),
}));
const clientId = requiredString(registration.client_id, "registration client_id");
const verifier = randomBytes(64).toString("base64url");
const challenge = createHash("sha256").update(verifier, "ascii").digest("base64url");
const authorization = {
  client_id: clientId,
  redirect_uri: "http://127.0.0.1/callback",
  response_type: "code",
  code_challenge: challenge,
  code_challenge_method: "S256",
  scope: "mcp:tools",
  state: randomBytes(18).toString("base64url"),
  resource: resourceUrl.href,
};
const approval = await fetch(new URL("/authorize", baseUrl), {
  method: "POST",
  redirect: "manual",
  headers: { "content-type": "application/x-www-form-urlencoded" },
  body: new URLSearchParams({ ...authorization, brunn_token: upstreamToken }),
});
assert.equal(approval.status, 302, `authorization returned HTTP ${approval.status}`);
const callback = new URL(requiredString(approval.headers.get("location"), "authorization redirect"));
assert.equal(callback.searchParams.get("state"), authorization.state);
const code = requiredString(callback.searchParams.get("code"), "authorization code");

const tokens = await json(await fetch(new URL("/token", baseUrl), {
  method: "POST",
  headers: { "content-type": "application/x-www-form-urlencoded" },
  body: new URLSearchParams({
    grant_type: "authorization_code",
    client_id: clientId,
    code,
    code_verifier: verifier,
    redirect_uri: authorization.redirect_uri,
    resource: resourceUrl.href,
  }),
}));
const accessToken = requiredString(tokens.access_token, "access token");

const transport = new StreamableHTTPClientTransport(resourceUrl, {
  requestInit: { headers: { authorization: `Bearer ${accessToken}` } },
});
const client = new Client({ name: `${label}-canary`, version: "1.0.0" });
try {
  await client.connect(transport);
  const listed = await client.listTools();
  const toolNames = listed.tools.map((tool) => tool.name).sort();
  const requiredToolNames = expectedRemoteToolNames([
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
  ], process.env);
  const missingToolNames = requiredToolNames.filter((name) => !toolNames.includes(name));
  assert.deepEqual(
    missingToolNames,
    [],
    `hosted MCP omits required tools: ${missingToolNames.join(", ")}`,
  );

  const opened = toolBody(await client.callTool({
    name: "memory.open",
    arguments: { task: `Verify hosted Brunn access for ${label}`, token_budget: 2_000 },
  }));
  const sessionId = findString(opened, "session_id");
  const content = [
    `# ${label} remote connector canary`,
    "",
    marker,
    "",
    `Verified through OAuth Streamable HTTP MCP at ${resourceUrl.href}.`,
    "",
  ].join("\n");
  const idempotencyKey = `remote-canary:${label}:${canaryPath}`.slice(0, 240);
  const written = toolBody(await client.callTool({
    name: "memory.write",
    arguments: {
      path: canaryPath,
      content,
      media_type: "text/markdown",
      idempotency_key: idempotencyKey,
    },
  }));
  const replayed = toolBody(await client.callTool({
    name: "memory.write",
    arguments: {
      path: canaryPath,
      content,
      media_type: "text/markdown",
      idempotency_key: idempotencyKey,
    },
  }));
  const read = toolBody(await client.callTool({
    name: "memory.read",
    arguments: {
      session_id: sessionId,
      requests: [{ path: canaryPath, view: "full", max_chars: 8_000 }],
    },
  }));
  assert.match(JSON.stringify(read), new RegExp(escapeRegExp(marker)));
  const briefingTopics = toolBody(await client.callTool({
    name: "briefing.topics",
    arguments: { session_id: sessionId },
  }));

  const binaryResults = process.env.BRUNN_UPLOAD_MANIFEST
    ? await archiveFiles(client, baseUrl, upstreamToken,
        JSON.parse(await readFile(process.env.BRUNN_UPLOAD_MANIFEST, "utf8"))
          .map((item) => ({...item,session_id:sessionId})))
    : [];

  process.stdout.write(`${JSON.stringify({
    label,
    resource: resourceUrl.href,
    tools: toolNames,
    session_id: sessionId,
    canary_path: canaryPath,
    write_status: findString(written, "status", false),
    write_reference: findString(written, "entry_ref", false)
      ?? findString(written, "reference", false),
    replay_status: findString(replayed, "status", false),
    briefing_topics_status: findString(briefingTopics, "status", false),
    marker_verified: true,
    verified_binaries: binaryResults,
  }, null, 2)}\n`);
} finally {
  await client.close().catch(() => undefined);
}

function toolBody(result) {
  assert.notEqual(result.isError, true, JSON.stringify(result.content));
  const first = Array.isArray(result.content) ? result.content[0] : undefined;
  assert.equal(first?.type, "text");
  return JSON.parse(first.text);
}

async function json(response) {
  const value = await response.json();
  assert.equal(response.ok, true, JSON.stringify(value));
  assert.ok(typeof value === "object" && value !== null);
  return value;
}

function findString(value, key, required = true) {
  if (Array.isArray(value)) {
    for (const item of value) {
      const found = findString(item, key, false);
      if (found !== undefined) return found;
    }
  } else if (typeof value === "object" && value !== null) {
    if (typeof value[key] === "string") return value[key];
    for (const item of Object.values(value)) {
      const found = findString(item, key, false);
      if (found !== undefined) return found;
    }
  }
  if (required) throw new Error(`missing ${key}`);
  return undefined;
}

function requiredString(value, labelName) {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`missing ${labelName}`);
  }
  return value;
}

function requiredEnvironment(name) {
  return requiredString(process.env[name], name);
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
