# Brunn hosted ChatGPT access

Status: production gateway live; ChatGPT Chat/Work account setup and mobile use
supported when the plugin is available to the account, 2026-08-27

Brunn exposes one authenticated Streamable HTTP MCP resource at:

```text
https://brunn.ai/mcp
```

The same hosted OAuth endpoint is used by ChatGPT, Codex, and Aether/OpenClaw.
The former local stdio launcher is retired for these clients.

## Trust and token model

The public web service proxies an exact allowlist of MCP and OAuth routes to a
private one-replica Railway service. The gateway implements OAuth 2.1
authorization code flow, S256 PKCE, dynamic client registration, RFC 8707
resource binding, and RFC 9728 protected-resource metadata.

The MCP route permits browser access only from the exact HTTPS origins in
`BRUNN_MCP_ALLOWED_ORIGINS`. Requests without an `Origin` remain valid for
non-browser MCP clients. Browser preflights run before bearer authentication,
and authenticated or 401 responses expose only the MCP session and OAuth
challenge headers required by browser clients. Protected-resource metadata is
public and remains readable from any origin.

Every product gets a distinct Brunn `read_write` credential scoped to
`scope:root`. The approval page rejects read-only credentials and owner tokens
with `admin` or `credential:manage`. It verifies a pasted credential through
`/v1/me`, does not persist or log it, and returns only encrypted gateway access
and refresh tokens to the connector. Revoking the dedicated Brunn
credential revokes its actual data access without affecting another client.

Each POST to the hosted `/mcp` resource receives an `X-Request-ID` and emits a
content-free outcome log. For mutation correlation the log contains only a
SHA-256 of a valid idempotency key, never the raw key, path, content, arguments,
credential, or token. This distinguishes a client-side approval drop (no
ingress record) from bearer, parser, dispatch, or upstream failures.

The persistent `BRUNN_MCP_SEALING_KEY` is a 32-byte random Railway secret.
Rotating it invalidates all remote registrations and sessions. Preserve it
across ordinary deployments.

## Hosted tool surface

The gateway exposes:

- `memory.open`, `memory.query`, `memory.read`, and `memory.changes`
- `memory.capture`, `memory.write`, and `memory.checkpoint`
- `memory.status`
- `asset.list`, `asset.metadata`, and `asset.upload_url`
- `briefing.publish`, `briefing.dedupe`, and `briefing.topics`
- `document.publish` and `document.get`
- `notification.publish`

It does not expose `memory.stage` or `asset.fetch`. Those operations read or
write the MCP adapter host's filesystem; in Railway that filesystem is not the
user's phone or computer. Text reads are capped at 120,000 characters per item
for hosted-client result limits. The local stdio adapter also exposes the two
filesystem-dependent tools.

### Binary uploads from a client with access to the file

Call `asset.upload_url` with `path`, `media_type`, `size_bytes`, and optionally
`sha256` and `expected_version`. PUT the raw bytes to the returned `put_url`
using **both returned headers** (`Authorization: BrunnUpload ...` and
`Content-Type`). This is a 15-minute, path/version-bound permission, not an API
bearer. The permission stays in a header, not URL/access logs. Never send file
bytes or base64 through MCP. Clients without filesystem/HTTP access must use
an upload-capable client; hosted MCP cannot read their attachments itself.

The existing streaming route and object store handle the bytes. A successful
PUT returns HTTP 201 with the actual entry reference, version, hash and size.
Retry an uncertain PUT with the same permission: HTTP 409 `upload_completed`
carries the published result in `error.details`. Read-only clients cannot mint.
An existing path requires its current version; omission is create-only. An
expired permission returns 410; size/hash mismatch returns 400; over 4 GiB
returns 413. Expiry is checked at the start of the request, not after transfer.

There is no upload table or duplicate result record. A signed grant reserves
the eventual immutable entry-version UUID, and publication under the existing
path lock checks that UUID plus the expected destination identity/version.
Replays cannot publish twice, even after rename or deletion. The signing uses
the existing server key with a dedicated JWT audience; it does not change MCP
OAuth. A grant remains valid until expiry even if its minting credential is
revoked meanwhile (maximum 15 minutes); account mutation locks still apply.

New hosted uploads create deterministic pending companions without invoking
the legacy paid description job. Dreamer filing and web/iOS pickers are
separate work. Credentialed binary GET remains available; hosted `asset.fetch`
and an agent-facing download grant are not introduced here.

## Railway deployment

The checked-in topology creates private service `mcp` from
`apps/mcp/Dockerfile.remote`, then passes its private hostname to the public
web proxy. Before applying topology changes, always run a plan and require the
diff to contain only the intended changes—no unrelated updates or deletions:

```bash
export RAILWAY_IAC_TS_BIN="$PWD/.railway/node_modules/.bin/railway-iac-ts"
railway config plan --verbose
railway config apply --yes --verbose
```

Do not apply a mixed-drift plan. Use a service-scoped variable update and
MCP-only deployment for a narrow gateway release, or reconcile the unrelated
topology drift as a separate change.

The service requires:

```text
PORT=8080
BRUNN_API_URL=http://api.railway.internal:8080
BRUNN_MCP_PUBLIC_URL=https://brunn.ai
BRUNN_MCP_ALLOWED_ORIGINS=https://chatgpt.com,https://brunn.ai
BRUNN_MCP_SEALING_KEY=<base64 for exactly 32 random bytes>
```

Deploy `mcp` before `web`, then verify health, discovery, an unauthenticated
OAuth challenge, a full authorization-code exchange, the complete hosted tool
surface, and a read/write canary using a dedicated client credential.

The checked-in canary consumes a credential only from its environment and does
not print or persist it:

```bash
cd apps/mcp
BRUNN_REMOTE_TOKEN="$(security find-generic-password -s brunn.ai -a CLIENT_ACCOUNT -w)" \
BRUNN_REMOTE_LABEL="client label" \
BRUNN_REMOTE_CANARY_PATH="operations/canaries/client.md" \
BRUNN_REMOTE_MARKER="UNIQUE_MARKER" \
node scripts/remote-canary.mjs
```

## Product setup

On ChatGPT web or desktop, enable Developer mode under **Settings → Security
and login**, then use the plus control on the Plugins page to register the full
MCP URL above. Complete OAuth with the dedicated ChatGPT credential and review
the discovered tools and metadata before enabling it. Developer mode can be
account- or policy-dependent.

This creates the cloud/account connection used by new Chat and Work
conversations. OpenAI documents account-available plugins as usable on mobile,
although web/desktop are the documented browse and install surfaces. On the
same account, open a new mobile Chat or Work conversation and select Brunn
from Plugins or `@` autocomplete. A local Codex stdio configuration or a local
plugin-creator personal-marketplace entry does not create this account-level
connection and is not a mobile provisioning path.

Current product references:

- [OpenAI plugin availability](https://learn.chatgpt.com/docs/plugins)
- [OpenAI ChatGPT connection setup](https://developers.openai.com/plugins/deploy/connect-chatgpt)
- [OpenAI MCP connections](https://learn.chatgpt.com/docs/extend/mcp)
- [OpenAI MCP authentication](https://developers.openai.com/plugins/build/auth)
