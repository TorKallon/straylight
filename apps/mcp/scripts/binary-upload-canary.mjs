// Optional real-file extension to the existing OAuth canary. The caller
// supplies an explicitly authorized manifest; no bytes or credentials logged.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";

export async function archiveFiles(client, baseUrl, upstreamToken, manifest) {
  const results = [];
  for (const item of manifest) {
    const info = await stat(item.local_path);
    assert.equal(info.size, item.size_bytes, `${item.path}: local size mismatch`);
    const digest = createHash("sha256");
    for await (const chunk of createReadStream(item.local_path)) digest.update(chunk);
    assert.equal(digest.digest("hex"), item.sha256, `${item.path}: local hash mismatch`);
    const minted = await client.callTool({name:"asset.upload_url", arguments:{
      path:item.path, media_type:item.media_type, size_bytes:item.size_bytes, sha256:item.sha256,
    }});
    assert.notEqual(minted.isError, true, `mint failed for ${item.path}`);
    const grant = JSON.parse(minted.content[0].text).data;
    assert.equal(new URL(grant.put_url).origin, baseUrl.origin);
    const response = await fetch(grant.put_url, {method:"PUT", headers:grant.headers,
      body:createReadStream(item.local_path), duplex:"half"});
    const publication = await response.json();
    assert.equal(response.status, 201, `${item.path}: PUT failed (${response.status}, ${publication.error?.code})`);
    const stored = publication.data;
    assert.equal(stored.content_hash, `sha256:${item.sha256}`);
    assert.equal(stored.size_bytes, item.size_bytes);
    const retry = await fetch(grant.put_url, {method:"PUT",headers:grant.headers,body:new Uint8Array()});
    const replay = await retry.json();
    assert.equal(retry.status,409);
    assert.equal(replay.error.code,"upload_completed");
    assert.equal(replay.error.details.entry_ref,stored.entry_ref);
    assert.equal(replay.error.details.version,stored.version);

    const metadata = await client.callTool({name:"asset.metadata", arguments:{
      session_id:item.session_id, asset_ref:stored.entry_ref, version:stored.version,
    }});
    assert.notEqual(metadata.isError,true);
    const exact = JSON.parse(metadata.content[0].text).data;
    assert.equal(exact.content_hash,`sha256:${item.sha256}`);
    assert.equal(exact.size_bytes,item.size_bytes);
    const download = await fetch(new URL(`/api/v1/workspace/binaries/${encodeURIComponent(stored.entry_ref)}/content?version=${stored.version}`,baseUrl),{
      headers:{Authorization:`Bearer ${upstreamToken}`},
    });
    assert.equal(download.status,200);
    let downloadedSize = 0;
    const downloadedHash = createHash("sha256");
    for await (const chunk of download.body) { downloadedSize += chunk.length; downloadedHash.update(chunk); }
    assert.equal(downloadedSize,item.size_bytes);
    assert.equal(downloadedHash.digest("hex"),item.sha256);
    const result = {...stored,original_filename:item.local_path.split("/").at(-1), stored_bytes_verified:true,replay_verified:true};
    results.push(result);
    process.stdout.write(`${JSON.stringify({verified_binary:result})}\n`);
  }
  return results;
}
