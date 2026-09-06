# Legal Research App

Local-first, project-based legal research and document intelligence system.
Upload documents, ask questions in plain language, and get an answer that's
cross-checked against the retrieved evidence rather than the model's own
recollection — with every claim traceable back to a specific chunk, page, or
(in Verified Research mode) an independently confirmed external source.

## Setup

```bash
# 1. Python deps (uv manages its own Python 3.12 install)
uv sync

# 2. Local Supabase (Postgres + pgvector + Auth + Storage), via Docker
supabase start
# copy the printed API_URL -> SUPABASE_URL, SERVICE_ROLE_KEY -> SUPABASE_KEY,
# and ANON_KEY -> SUPABASE_ANON_KEY into .env

# 3. Copy env template and fill in whichever provider keys you have
cp .env.example .env

# 4. Run tests
uv run pytest -q

# 5. Run the app
uv run streamlit run src/legal_research_app/ui/app.py
```

`supabase stop` shuts the local stack down. For schema changes, use
`supabase migration up` (applies only new migrations) rather than
`supabase db reset` once the database holds any real documents —
`db reset` recreates the database from scratch and destroys all data, not
just schema.

## Architecture

**Providers.** The active profile (`ACTIVE_PROFILE` in `.env`) assigns a
provider+model to every task (query expansion, embeddings, evidence
synthesis, verification) in one place (`src/legal_research_app/config.py`).
The app ships hardcoded to OpenAI for reasoning and a local BGE-M3 model
(`sentence-transformers`) for embeddings, so ingestion costs no API calls;
the underlying provider registry also supports Gemini and Ollama for a fully
local/offline profile. There is no silent fallback — if a task's configured
provider is unavailable, or a privacy mode forbids it, the call raises a
clear error rather than switching providers behind your back
(`providers/registry.py`).

**Ingestion.** PDFs are parsed with Docling (headings, page numbers, and
tables preserved; OCR available for scanned documents), then chunked so
whole sections stay together and only oversized sections get sub-split with
overlap. Multiple files can be uploaded and ingested in one action.

**Retrieval.** Dense (pgvector cosine similarity) and lexical (Postgres
full-text, `websearch_to_tsquery` — quoted phrases, `OR`, `-exclusion`
supported natively) search run independently and are combined with
Reciprocal Rank Fusion, since the two scores live on incomparable scales.
Query expansion (`SynonimiseProvider`) generates alternate phrasings and
decomposes them into individual significant words, so a question that shares
no exact terminology with a document can still surface it lexically.

**Research modes.**
- **Simple** — one pass: semantic search → query expansion → keyword search →
  fuse both result sets → one LLM call that answers the question and is
  explicitly instructed to flag or drop anything the retrieved excerpts don't
  support. Fast, single-shot.
- **Advanced (Quick / Standard / Deep)** — a multi-round agent
  (plan → generate queries → search → synthesize → challenge → gap analysis →
  loop or finalize), with deterministic per-mode round limits and stopping
  based on retrieval novelty. Produces a full report: executive answer,
  confidence (from observable signals — evidence quantity, source diversity,
  contradiction status, subquestion coverage — never self-reported by the
  model), key findings with their cited evidence, contradictions, gaps, and
  research coverage. State is persisted after every step, so a run can be
  interrupted and resumed.
- **Verified Research** — Deep mode plus a second evidence channel that
  searches the live web, restricted to an allowlist of primary Indian legal
  sources (indiankanoon.org, sci.gov.in, any `.gov.in`/`.nic.in` domain) and a
  few established secondary ones. Every piece of evidence is tagged
  `user_document` / `primary` / `secondary`, and a primary source is weighted
  above a secondary one when they conflict. Case names the model recalls are
  treated as unverified leads and independently confirmed (via search plus
  title matching) before they're allowed to stand as evidence — anything that
  can't be confirmed is surfaced explicitly as an unconfirmed citation rather
  than silently trusted.

**Anti-fabrication.** The model is never allowed to write a chunk ID, page
number, or filename itself. Every synthesis prompt shows it a numbered
evidence list and it can only cite by index; every parser validates those
indices against the real range and drops anything out of range.

**Search Builder.** A direct, low-level interface to the retrieval layer for
inspecting result quality without running the full agent — dense, lexical,
or fused search, plus a corpus-derived term-suggestion tool: real phrases
that recur across a query's nearest-neighbor chunks, surfaced as possible
additional search terms. This is deliberately not LLM-generated — a term
only appears if it's real text found independently in more than one matched
chunk.

## Known limitations

- **Numeric/threshold boundary reasoning can be unreliable.** When a
  question hinges on a precise quantitative comparison against a legal
  threshold (e.g. an operator like "greater than" vs. "at least"), the
  synthesis step can misapply a correctly-retrieved rule, even when that
  rule is stated verbatim in the evidence shown to it. This has persisted
  across multiple prompt-engineering attempts and more than one model, and
  is a known reliability gap rather than a solved problem. Treat any answer
  that turns on an exact boundary value with extra scrutiny, and verify
  independently before relying on it.
- **Rare-fact retrieval in large, similar-document corpora.** A single-pass
  Simple-mode search can fail to surface one specific, rarely-stated fact
  buried in a large set of lexically/semantically similar documents (e.g.
  many court orders sharing the same boilerplate). Advanced/Deep mode's
  multi-round, higher-volume search is more reliable for precise fact-lookup
  questions against a large document set. A cross-encoder reranker would
  likely close this gap further and is scoped but not yet implemented.
- Research runs block synchronously in the UI (no async/websocket layer
  yet); a Deep or Verified run is felt as UI wait time for its full duration.
- PDF only for now — DOCX/TXT/HTML ingestion isn't implemented.
- No authentication in the current build — the app runs against a single
  implicit workspace. A real Auth implementation (Supabase Auth + RLS) has
  been built and verified against this schema before, but is not wired into
  the current UI.

## Not yet implemented

- DOCX/TXT/HTML/other format ingestion; OCR path untested
- Cross-encoder reranking
- A CLI (Streamlit UI exists; no CLI entry point yet)
- Async/background research execution and mid-run pause/resume from the UI
- Normalized findings/evidence tables (currently jsonb on `research_sessions`
  — no cross-session querying need has appeared yet)
