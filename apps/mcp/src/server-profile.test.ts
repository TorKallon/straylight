import assert from "node:assert/strict";
import test from "node:test";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

import { BrunnApiClient } from "./api-client.js";
import { createBrunnMcpServer } from "./index.js";

test("remote profile exposes only hosted-safe tools with bounded reads", async () => {
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const apiClient = new BrunnApiClient(
    "https://api.invalid",
    "test-token",
    async () => new Response(JSON.stringify({ status: "ok" }), {
      status: 200,
      headers: { "content-type": "application/json" },
    }),
  );
  const server = createBrunnMcpServer(apiClient, {
    surface: "remote",
    includeStructuredContent: true,
  });
  const client = new Client({ name: "remote-profile-test", version: "0.1.0" });

  try {
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    assert.equal(client.getServerVersion()?.name, "Brunn");
    assert.match(client.getInstructions() ?? "", /Start substantive work with memory\.open/);
    assert.match(client.getInstructions() ?? "", /memory\.checkpoint/);
    const response = await client.listTools();
    const expectedNames = [
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
    for (const name of expectedNames) {
      assert.ok(response.tools.some((tool) => tool.name === name), `missing hosted tool ${name}`);
    }
    assert.equal(response.tools.some((tool) => tool.name === "memory.stage"), false);
    assert.equal(response.tools.some((tool) => tool.name === "asset.fetch"), false);

    const read = response.tools.find((tool) => tool.name === "memory.read");
    assert.ok(read);
    const requests = read.inputSchema.properties?.requests as {
      items?: { properties?: { max_chars?: { maximum?: number } } };
    } | undefined;
    assert.equal(requests?.items?.properties?.max_chars?.maximum, 120_000);

    const write = response.tools.find((tool) => tool.name === "memory.write");
    const upload = response.tools.find((tool) => tool.name === "asset.upload_url");
    assert.equal(upload?.annotations?.readOnlyHint, false);
    assert.match(upload?.description ?? "", /returned headers/);
    assert.match(upload?.description ?? "", /Never put file bytes or base64/);
    const query = response.tools.find((tool) => tool.name === "memory.query");
    const checkpoint = response.tools.find((tool) => tool.name === "memory.checkpoint");
    assert.equal(write?.annotations?.readOnlyHint, false);
    assert.equal(write?.annotations?.destructiveHint, false);
    assert.equal(write?.annotations?.idempotentHint, false);
    assert.match(write?.description ?? "", /retry at most once/);
    assert.match(write?.description ?? "", /every argument and the key unchanged/);
    assert.equal(checkpoint?.annotations?.readOnlyHint, false);
    assert.equal(checkpoint?.annotations?.idempotentHint, true);
    assert.equal(query?.annotations?.readOnlyHint, true);
    assert.equal(query?.annotations?.openWorldHint, false);
  } finally {
    await client.close().catch(() => undefined);
    await server.close().catch(() => undefined);
  }
});

test("hosted upload tool forwards only upload metadata and preserves returned PUT headers", async () => {
  const calls: Array<{ url: string; method: string | undefined; body: unknown }> = [];
  const api = new BrunnApiClient("https://api.invalid", "test-token", async (url, init) => {
    calls.push({url: String(url), method: init?.method, body: JSON.parse(String(init?.body))});
    return new Response(JSON.stringify({status:"complete", data:{put_url:"https://brunn.ai/api/v1/workspace/binaries/content", headers:{Authorization:"BrunnUpload fixture", "Content-Type":"image/jpeg"}}}), {status:200, headers:{"content-type":"application/json"}});
  });
  const server = createBrunnMcpServer(api, {surface:"remote", includeStructuredContent:true});
  const client = new Client({name:"binary-upload-test",version:"1"});
  const [a,b] = InMemoryTransport.createLinkedPair();
  try {
    await server.connect(b); await client.connect(a);
    const input = {path:"Inbox/example.jpg",media_type:"image/jpeg",size_bytes:123,sha256:"a".repeat(64),expected_version:2};
    const result = await client.callTool({name:"asset.upload_url",arguments:input});
    assert.notEqual(result.isError,true);
    assert.deepEqual(calls,[{url:"https://api.invalid/v1/uploads",method:"POST",body:input}]);
    assert.match(JSON.stringify(result.content),/BrunnUpload fixture/);
  } finally { await client.close(); await server.close(); }
});
