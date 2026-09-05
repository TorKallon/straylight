# AGENTS.md

Brunn is the durable memory, evidence, checkpoint, and continuity system
on Nyx. Read `README.md`, `docs/Architecture.md`, `docs/Specification.md`, and
`docs/Operations.md` as needed; current code and tests are implementation
truth.

## Design system

All user-facing visual work — the web SPA, the iOS app, and any future
surface — must follow `docs/Brand.md` (the **Still Water** design system).
In practice:

- Use the defined tokens (CSS custom properties in `apps/web/src/styles.css`,
  `BrunnTheme` in `apps/ios`), never new color literals.
- The web app is dark by default (`data-theme="light"` restores light); any
  new foreground/background pairing must be WCAG-validated in both
  appearances, and chart series colors must come from the validated data
  palette.
- Do not reintroduce retired forest-green-era colors, and never use the brand
  blue to mean "success".

## Code retrieval

Before adding a helper or duplicating an existing behavior:

1. Use `rg` for exact names, strings, routes, SQL, config keys, and errors.
2. Use `nyx-code-intel search --project brunn "<intent>"` when the name is
   unknown.
3. Use `nyx-code-intel lsp --project brunn ...` for precise symbols,
   definitions, and references.

Use `--semantic=false` when an entirely local search is preferred. The index
still returns exact and FTS results if OpenAI semantic search is unavailable.

Use Codex Session Search for historical commands, failures, decisions, and
handoffs. Use Brunn memory for durable conclusions, checkpoints,
provenance, and cross-session continuity. Do not ingest full source trees into
Brunn; the local code index is authoritative for the current checkout.

The shared implementation and runbook live at:

```text
/Users/Shared/projects/metis/docs/nyx-code-intel.md
```

## Model billing

All agent reasoning, evaluation, grading, and benchmark runs must use the
owner's ChatGPT-authenticated Codex plan. Never switch or fall back to OpenAI
API-key billing when Codex limits are reached. Stop the run so the owner can
switch ChatGPT accounts or wait for the plan reset.

OpenAI API billing is allowed only for explicitly approved product capabilities
that the Codex plan cannot provide as an API, such as text embeddings. The
reasoning harness must remove API credentials and routing overrides from child
processes and fail closed unless `codex login status` reports `Logged in using
ChatGPT`.

## Verification

Hosted binary intake uses `asset.upload_url`, then an HTTP PUT of raw bytes
with the returned headers. Never inline binary/base64 into a tool call or use
the retired local launcher. Reuse the same permission after an uncertain PUT;
`upload_completed` returns the existing publication, not another version.
Discover required hosted tools by name, never by an exact tool count.

Run the narrowest checks that cover the change. Common gates are:

```bash
python3 -m unittest discover -s tests -v
(cd apps/mcp && npm run build && npm test)
(cd apps/web && npm run build && npm test -- --run)
```

## Location

Location: treat `low` confidence as a hint and `stale` as "last known". Never ask the owner where he is when owner_presence is present. In shared or group contexts do not reveal location unless the owner asks. A night is away when midnight falls between a Home departure and the next Home arrival. After editing Location/Places.md, run location.rederive for the affected window. Places where the owner moves continuously (resorts, tracks, parks, campuses) must be known places in Location/Places.md.
