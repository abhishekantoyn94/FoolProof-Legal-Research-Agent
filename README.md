# Legal Research App

Local-first, provider-agnostic legal research system. See the project-level
plan for full architecture context; this README covers setup for what exists
so far (Phase 1 — Foundation).

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

# 5. Run the (throwaway internal) UI
uv run streamlit run src/legal_research_app/ui/app.py
```

`supabase stop` shuts the local stack down; `supabase db reset` re-applies all
migrations from scratch against a clean local database.

## Provider profiles

The active profile (`ACTIVE_PROFILE` in `.env`) assigns a provider+model to
every task (planner, query expansion, embeddings, reranker, evidence analysis,
verification, final synthesis) in one place — see `src/legal_research_app/config.py`.

| Profile | Privacy mode | Reasoning tasks | Embeddings |
|---|---|---|---|
| `local_only` | LOCAL_ONLY | Ollama (`gemma4:e2b-mlx`) | local BGE-M3 |
| `hybrid_openai_default` (default) | HYBRID | OpenAI (`gpt-4o-mini`) | local BGE-M3 |
| `hybrid_openai_gemini_verify` | HYBRID | OpenAI, Gemini for verification only | local BGE-M3 |

**No silent fallback**: if a profile's chosen provider is unavailable or its
privacy mode forbids a task's configured provider, the call raises a clear
error (`ProviderError` / `PrivacyViolationError`) rather than switching
providers on your behalf. See `src/legal_research_app/providers/registry.py`.

## What's implemented (Phase 6)

A Streamlit internal validation UI (`ui/app.py`) — deliberately throwaway per
the agreed sequencing (engine first, real Next.js + auth UI later). No login:
an org picker in the sidebar stands in for real multi-tenancy until Phase 7.

- Sidebar: provider profile selector, a highly visible privacy-mode banner
  (section 19), provider health (✅/❌ per provider), and a read-only model
  control panel showing the active profile's task→provider/model assignments
- **Collections tab**: create/list Projects and Knowledge Base categories
- **Documents tab**: upload a PDF, ingest it (parse → chunk → embed →
  persist) with a spinner and honest status/error reporting, document library
  listing with status icons
- **Research tab**: pick a Project and/or KB categories as retrieval scope,
  jurisdiction, research mode, ask a question, run it, and see the full
  section-60 answer format: status, operational trace, executive answer,
  confidence (with rationale), key findings (expandable to their cited
  evidence), contradictions, gaps, research coverage, and the full evidence list
- **History tab**: reopen a past session's stored answer without re-running it

**Verified in an actual browser, not just backend tests**: launched the app,
created an organization and project, uploaded the synthetic fixture PDF
through the real file-upload control (indexed: 4 chunks, 2 pages, rendered
live in the document library), ran a QUICK-mode research question through the
UI, and confirmed the full answer rendered correctly — 4 findings with real
page/heading citations, gaps, verifier notes, research coverage, and MEDIUM
confidence, matching what the Phase 5 backend tests already established.

**One real bug caught in this pass**: the organization picker used a
name-keyed dict (`{name: id}`) while every other picker in the same file
(projects, categories, documents) already used the safer pattern of selecting
the object directly. An accidental double-click while testing created two
identically-named orgs, which exposed it immediately: Postgres doesn't
guarantee row order for ties beyond the explicit sort key, so the dict could
silently resolve "Acme Legal" to a different org_id across reruns, making
newly-created projects appear to vanish. Fixed by switching the org picker to
the same select-the-object pattern already used everywhere else.

Known limitation: research runs block synchronously in the UI (no
async/websocket layer yet), so QUICK/STANDARD/DEEP mode durations with a local
model are fully felt as UI wait time (~100-200s locally; much faster with a
cloud provider). Mid-run pause/resume from the UI itself isn't wired up yet,
though the underlying agent supports it (Phase 5).

## Incident: local database wiped, recovered

While applying the Verified Research migration, `supabase db reset` was run
without accounting for the fact that real ingested documents existed in the
local database by this point -- that command recreates the database from
migrations, destroying all data, not just schema. All document/chunk/session
rows were lost. Recovered: the underlying PDF *files* survive in a separate
Docker volume (Supabase Storage keeps file bytes separately from the Postgres
metadata that references them), so all 12 real documents were extracted
directly from that volume and re-ingested into a new project. Going forward,
`supabase migration up` (applies only new migrations, no data loss) is used
instead of `db reset` for any schema change now that real data exists.

## Verified Research: web-search + source-authority engine

Added a new `ResearchMode.VERIFIED`: the full DEEP-style multi-round agent
(contradiction detection, gap analysis, iterative search) plus a second
evidence channel that searches the live web, restricted to an allowlist of
primary Indian legal sources (indiankanoon.org, sci.gov.in, any `.gov.in`/
`.nic.in` domain) plus a few established secondary ones (LiveLaw, SCC Online,
Bar & Bench). Every piece of evidence -- local document or web result -- is
tagged `user_document` / `primary` / `secondary`, and the synthesis/challenge
prompts are explicitly instructed that a primary source outweighs a secondary
one when they conflict, and that a finding resting only on a secondary source
must say so rather than presenting itself as settled.

Built with Tavily (`providers/tavily_provider.py`) as the search backend --
chosen because it's purpose-built for LLM/agent research (native domain
restriction, optional full-page content in one call) rather than a bare
search-snippets API needing a separate fetch step.

**Real bug found and fixed via live testing**: Tavily's own `include_domains`
parameter is *not* a strict filter -- confirmed empirically, a 7-domain
allowlist still returned a result from `supremetoday.ai`, which wasn't on it.
Fixed by never trusting the external API's restriction to be authoritative:
`TavilyWebSearchProvider` now re-filters every result against the allowlist
itself (`_domain_allowed`), regression-tested with a mocked response proving
the filter actually drops disallowed domains regardless of what the live API
does on any given day.

**Verified end-to-end for real**, against the actual question that started
this: "if arrested with exactly 20kg of Ganja, does Section 37 apply". The
live run pulled in real evidence including a genuine Andhra Pradesh High
Court ruling ("6 Kgs Ganja Not Commercial Quantity, Rigours Of Bail U/S 37
NDPS Act Not Applicable") and a real Supreme Court order PDF from
api.sci.gov.in, both correctly tagged `primary`, alongside the local
documents (tagged `user_document`). The agent detected a genuine
contradiction between the bare statutory "greater than" language and how a
court applied it to a different substance, and rather than confidently
picking a side, it correctly reported **low confidence** with the
contradiction stated explicitly -- "the application... remains contested and
incomplete" -- which is the right answer for a genuinely unresolved boundary
question, and a direct improvement over Simple mode's earlier confidently
wrong answer to the same question.

**Known real limitation, not yet solved**: the synthesis didn't clearly
connect the highly relevant AP High Court "6kg" precedent to the specific
"exactly 20kg" question in its final prose -- the right primary source was
retrieved and tagged correctly, but drawing the precise analogy in the answer
itself is still inconsistent. This is the same class of multi-hop reasoning
gap found earlier in this session, now recurring one level up (across
sources, not just across substances within one document). A stronger
synthesis-task model, or a dedicated case-law-analogy step, would likely help
more than further prompt tuning on `gpt-4o-mini`.

## Real-world usage findings (your own NDPS Act documents)

Testing against your actual uploaded documents (not synthetic fixtures)
surfaced three genuine issues, in increasing order of depth:

1. **Fixed: Storage `InvalidKey` on filenames with special characters.** A
   real filename ("...— 2026 - Bhatt & Joshi Associates.pdf") has an em-dash,
   which Supabase Storage rejects as an object key character (400
   `InvalidKey`) -- `DocumentService` used the raw filename directly as the
   storage path. Fixed with `_safe_storage_key_component()`: the storage key
   is sanitized, the human-readable `filename` shown in the UI is untouched.
   Regression-tested with the exact problematic filename pattern.

2. **Fixed: a genuine reasoning gap, not a retrieval gap.** Asked whether
   exactly 20kg of Ganja triggers Section 37's bail rigour, the model
   answered confidently wrong in both Simple and Deep mode. The correct
   principle *was* in the retrieved evidence -- a court order excerpt stating
   "commercial quantity" means a quantity **greater than** (not equal to) the
   notified threshold, illustrated with Methamphetamine/50g. The model:
   (a) treated the principle as scoped to Methamphetamine only rather than
   applying it to Ganja's own threshold, and (b) supplied the specific 20kg
   figure from its own training data despite being told not to, then
   compared it to itself and got the boundary comparison wrong. Fixed with
   two additions to every evidence-grounding prompt (`agent/prompts.py`):
   `LEGAL_PRINCIPLE_APPLICATION_INSTRUCTION` (apply stated principles across
   differing facts, don't scope them to the illustrating example) and an
   explicit instruction to state plainly when a specific figure needed to
   answer isn't in the excerpts, rather than filling it in from training
   knowledge. Verified: the model now correctly says it cannot conclude
   without the specific threshold, instead of confidently asserting the
   wrong one.

3. **Real, deeper limitation surfaced, not fully solved: rare-fact retrieval
   in a large corpus of similar documents.** Even after (2), Simple mode's
   single-round retrieval couldn't reliably surface the one chunk stating
   Ganja's actual 20kg threshold, buried in ~12 real documents full of
   similar court-order boilerplate ("O R D E R", "arrested", "bail", "section")
   that lexically/semantically dilute the one rare, specific fact. Traced
   precisely: the crucial chunk ranks 45th in the fused candidate list even
   at a 40-chunk pool -- a genuine RRF/keyword-dilution effect, not a bug
   with an easy one-line fix. Along the way, found and fixed a real
   contributing bug: `build_websearch_query` decomposed multi-word
   Synonimise *expansions* into individual words (the Phase 4 fix) but never
   applied that same decomposition to the **original question itself**, so
   an entity named directly in the question ("Ganja") wasn't guaranteed to
   survive as its own searchable term if Synonimise's paraphrases happened to
   drop it (e.g. into "cannabis", "narcotic substance"). Fixed and
   regression-tested (`test_build_websearch_query_decomposes_the_original_question_too`).
   This measurably helps but does not fully close the gap on a large corpus.
   **Current guidance**: for precise fact-lookup questions against a large
   document set, use Advanced/DEEP mode (confirmed to retrieve the correct
   chunk via its multi-round, higher-volume search) rather than Simple mode.
   The proper long-term fix is a reranker (cross-encoder re-scoring of
   candidates), which is scoped but not yet implemented.

Also noticed, not yet acted on: you have two identical copies of "Exact
marginal commercial quantiry.pdf" ingested under separate document rows --
worth deleting one via the Documents page, since duplicate identical sources
can artificially inflate "multi-source" confidence scoring.

## Prototype: corpus-derived term suggestion (Search Builder)

Idea explored: could vector-space geometry suggest better search keywords,
instead of relying on an LLM (Synonimise) to invent them? A generic
word-embedding nearest-neighbor lookup wasn't the right fit -- BGE-M3 is
trained to place similar *passages* near each other, not to act as a
thesaurus for individual words. The practical version that's actually buildable
with existing infrastructure: run the query through the *existing*
`dense_search`, then extract phrases that genuinely **recur across multiple**
of the returned chunks. A term only surfaces if it's real corpus text found
independently in more than one match -- a cheap, built-in anti-fabrication
property, since nothing here is generated.

Implementation: `retrieval/neighbor_term_extraction.py`
(`extract_salient_neighbor_terms`), wired up as `SearchService.suggest_terms()`
and exposed as a "💡 Suggest terms from corpus" button on the Search Builder
page.

**Finding from testing against the real recovered NDPS corpus**: the first
version (1-3 word n-grams) surfaced mostly noise -- `bail`, `accused`, `Act`,
`offence`, `Court` -- because single common legal words trivially recur across
nearly every chunk in a legal corpus; they're not stopwords generically, but
they behave like stopwords *in this domain*. Restricting to 2-3 word phrases
(dropping single words from the default) fixed this immediately: the same
query ("does section 37 bail rigor apply when exactly 20kg of ganja is
seized") now surfaces `commercial quantity`, `Section 37`,
`Psychotropic Substances Act`, `bail application`, `State of Kerala`, and even
a real case citation fragment (`2024 KER 50376`) pulled straight from the
corpus. Confirmed working end-to-end in the browser, not just unit tests.

Status: validated prototype, not yet wired back into the Synonimise
query-expansion pipeline itself -- that integration is a separate decision,
deliberately deferred until this stands on its own as useful.

## Simplification pass: core engine focus

Per direction, auth/multi-org/Knowledge-Base-categories were dropped from the
UI to refocus on the core engine, and a new **Simple** research mode was
added as the primary experience, with OpenAI as the only provider surfaced.

- **No login, single implicit workspace.** `ui/auth.py`, the auth RPCs, and
  RLS policies are left in the codebase/schema unused (easy to re-enable
  later) -- the app just auto-creates/reuses one "Default Workspace" org via
  the service-role client, same as before Phase 7.
- **Projects only** -- Knowledge Base categories are gone from the UI
  (`kb_categories` table stays in the schema, unused). A **"+ New Project"**
  button in the sidebar creates "Project N" immediately, with a rename
  popover -- matches the described flow directly (no upfront naming form).
- **Multi-file upload** -- `page_documents` now accepts and ingests several
  PDFs in one action (`st.file_uploader(..., accept_multiple_files=True)`),
  not just one at a time.
- **New SIMPLE research mode** (`ResearchMode.SIMPLE`, migration
  `20260906000005`): the core engine loop as described --
  1. semantic (dense) search on the raw question
  2. Synonimise keyword/phrase expansion
  3. keyword (lexical) search on the expansion
  4. fuse both result sets (Reciprocal Rank Fusion, reused from Phase 3)
  5. one OpenAI call that answers the question **and cross-checks itself**
     against the retrieved chunks (explicitly instructed to flag or drop
     anything the excerpts don't support)

  Implemented as `ResearchAgent.run_simple()` -- a standalone method, not
  routed through the Phase 5 state machine, since it doesn't loop or need
  pause/resume. Reuses `EvidenceItem`/`Finding`/`assess_confidence`/anti-
  fabrication index validation from the existing agent code rather than
  duplicating it. QUICK/STANDARD/DEEP (the fuller iterative agent --
  contradiction detection, gap analysis, multi-round loop) remain available
  under an "Advanced" toggle on the Research page, per your call to keep them.
- **OpenAI-only surface**: Settings no longer exposes provider-profile
  switching; the app is hardcoded to the `hybrid_openai_default` profile
  (OpenAI for all reasoning tasks, local BGE-M3 for embeddings so ingestion
  costs no API calls). The multi-provider registry itself wasn't ripped out
  -- just not surfaced in this UI.

**Verified end-to-end in the real browser**: no login screen; created a
project via the button; uploaded *two* PDFs at once (one an amendment
contradicting the other) in a single action, both indexed correctly; asked
"Is there any discrepancy in the settlement payment amount between the
documents?" in Simple mode and got a correct, HIGH-confidence answer citing
both documents and correctly stating the $500,000 vs $750,000 discrepancy --
in one pass, no multi-round loop.

**One real bug caught and fixed during this pass, but it was in my test
harness, not the product**: while manually constructing a multi-file browser
upload test, I mislabeled which base64 blob belonged to which filename
(mixed up a value read fresh with one from earlier in the session), which
made the two documents' `filename` fields look swapped in the UI. Reproduced
the exact two-file ingestion directly against `DocumentService` outside the
browser to confirm the real pipeline was unaffected (it was correct), then
fixed the mislabeled test data.

## What's implemented (Phase 7)

Real multi-user Supabase Auth, replacing the Phase 6 org-picker stand-in.

- Sign-up / sign-in / sign-out in the Streamlit app; every table operation
  now runs through a **per-user, JWT-authenticated client** (`db` in
  `ui/app.py`), not the service-role client -- RLS (written back in Phase 1)
  is what actually enforces org isolation now, for real, not just in theory
- Two security-definer RPCs (same pattern as Phase 1's `is_org_member`):
  `create_organization_with_owner` (atomic org + owner-membership creation)
  and `invite_member_to_org` (owner/admin adds an already-registered user by
  email, role-checked). A third, `list_org_members`, supports the invite UI.
  Organizations and org_members have **no direct INSERT policy** -- creation
  and membership changes are only possible through these vetted RPCs
- **Storage stays a service-role operation**: Supabase Storage has its own
  policy layer separate from table RLS, with no grant for `authenticated` in
  this schema. Confirmed empirically (a real user JWT gets a 403 on upload)
  rather than assumed. `DocumentService` now takes an optional `storage_db`
  (defaults to `db` if not given, so every Phase 2 test needed zero changes);
  the Streamlit app passes the user client for `db` and the service-role
  client for `storage_db`
- **One real RLS gap found and fixed while wiring this up**: `document_chunks`
  only ever had a SELECT policy (fine when only the service-role client
  existed). A real per-user client would have been silently blocked from
  writing chunks during ingestion. Brought in line with every other
  tenant-scoped table (`for all`, scoped to org membership) and added a
  regression test for exactly this case
- Minimal team invites: an org owner/admin can add an existing signed-up user
  by email; the Settings page shows current members and, for owners/admins,
  an invite form

**Verified for real, not assumed from reading the SQL**: signed up two users,
confirmed a brand-new user sees zero organizations, created an org via the
RPC and confirmed ownership + visibility, confirmed a *different* user sees
nothing until invited, confirmed a plain member is rejected when trying to
invite someone else, confirmed inviting a non-existent email fails clearly,
and confirmed a real per-user client can now write `document_chunks` (the
regression test for the fix above). Then walked the entire flow in an actual
browser: sign up → create org → create project → upload the fixture PDF
(ingested for real under the user's own RLS-scoped session, with storage
handled by service-role) → Settings → invite a second real user → confirmed
they appear in the members list → signed out → signed back in successfully.

One cosmetic-only quirk noted, not fixed: after sign-out, Streamlit's
native nav sidebar visually persists the page list from the prior session
until you click something; clicking correctly re-shows the sign-in gate with
no data exposed (verified), so this is not a security issue -- just a rough
edge in Streamlit's own multipage nav chrome.

## UI restructuring: "Foolproof AI"

The Streamlit app was renamed and restructured into a left-nav product shape
per a reference layout: **Research / Documents / Collections / Search Builder
/ Research History / Settings**, using Streamlit's native `st.navigation`.
This was a pure UI-layer change (no backend/service/schema changes) --
Settings now holds what used to be loose sidebar controls (provider profile,
privacy banner, provider health, model assignments); **Search Builder** is
the one new page, a thin debug UI directly over the already-tested
`SearchService` (dense/lexical/hybrid, with scope filters) for inspecting
retrieval quality without running the full agent. "Saved Research" and
"Chat" from the reference layout were deliberately left out per your call --
History covers the former, and Chat is a distinct feature not yet built.

**Two more real bugs caught while testing this in-browser (not hypothetical):**
1. `st.switch_page` (used by History's "Open in Research" button) was
   observed to reset the organization selectbox's own widget state back to
   "(create new)", silently dropping the active workspace. Fixed by tracking
   the selected org in an explicit `session_state["org_id"]` key and deriving
   the selectbox's `index` from that every rerun, rather than trusting the
   widget's own state to survive a page switch.
2. (Carried over from initial Phase 6 testing, documented below.)

Verified again end-to-end in the real browser after both fixes: uploaded
document still visible across pages, a real Search Builder query against the
live index returned correct fused results, and reopening a completed research
session from History via "Open in Research" correctly preserved the
workspace and loaded the stored answer.

## What's implemented (Phase 5)

The research agent state machine (master prompt section 35):
`IDLE → UNDERSTAND_AND_PLAN → GENERATE_QUERIES → SEARCH → SYNTHESIZE →
CHALLENGE → GAP_ANALYSIS → (loop, or) → VERIFY → FINAL`, persisted to
`research_sessions` after every transition.

- **Anti-fabrication (section 31)**: the LLM never writes a chunk_id, page
  number, or filename. Every synthesis/challenge prompt shows it a numbered
  evidence list; it can only cite index numbers, and every parser
  (`agent/prompts.py`) validates those indices against the real range and
  silently drops (with a logged warning) anything out of range. Verified for
  real: a finding citing a fabricated index is dropped; a finding left with
  zero real evidence doesn't survive at all.
- **Deterministic research depth (section 33)**: QUICK/STANDARD/DEEP have
  fixed min/max search-round counts (`agent/types.py`); the LLM's gap analysis
  can request more rounds within that range but never fewer than the mode's
  baseline, and stops early if retrieval novelty drops below threshold
  (diminishing returns, section 12). This decision is a pure function
  (`agent/control.py`) with no LLM or DB dependency, so it's fully unit-tested.
- **Confidence from observable signals only (section 61)**: evidence quantity,
  source diversity, dual-retrieval agreement, contradiction status, and
  subquestion coverage -- never "ask the LLM how confident it is"
  (`agent/confidence.py`).
- **Interruptible/resumable (section 36)**: `stop_requested` is checked at
  every state-machine checkpoint, including before the very first (planning)
  step; state (`agent_state` + `state_data`, the full working memory: plan,
  evidence, findings, contradictions) is persisted as jsonb so a paused or
  crashed session resumes exactly where it left off.
- `ResearchService.start_research` / `.resume` / `.request_stop` -- the entry
  point a CLI/UI calls.

**Two real bugs the tests caught (not hypothetical -- both found by running
against real infra, not by inspection):**
1. My first stopping-rule design used one shared "extra rounds" constant for
   every mode. A unit test (`test_agent_control.py`) immediately showed QUICK
   mode would loop anyway whenever the LLM reported any gap, defeating the
   point of a "quick" mode. Fixed by giving each mode its own explicit
   min/max round bounds.
2. The interruptibility check cleared `stop_requested` in the database
   *before* the loop got a chance to act on it, so a stop requested against a
   non-IDLE session was silently swallowed -- the loop's own check would
   always see "already cleared" and run one more step regardless. Fixed with
   a single atomic check-and-clear used consistently at every checkpoint.
   Caught by `test_research_agent_interruptibility.py` using a `SearchService`
   stub that fails the test loudly if the agent ever reaches it.

**Verified end-to-end against live infra** (not mocked): a full QUICK-mode run
across two synthetic documents (one an amendment contradicting the other) --
plan created, queries generated via Synonimise, hybrid search, synthesis,
challenge, verification, final synthesis, all persisted correctly, with every
cited evidence chunk_id confirmed to exist in the real database. A separate
DEEP-mode run confirmed the minimum-3-rounds guarantee holds for real (not
just in the pure unit test). Total real local-LLM run time: QUICK ~100s, DEEP ~180s.

## What's implemented (Phase 4)

- `SynonimiseProvider`: rebuilt from `Legal-AI-Agents-Local-`'s `synonimize()`
  as a provider-agnostic `QueryExpansionProvider` -- wraps whatever
  `LLMProvider` the active profile assigns to `query_expansion`, no per-vendor
  code. Jurisdiction is a parameter, not hardcoded; the old Indian-Kanoon-only
  `ORR`/`ANDD`/`NOTT` operators are gone -- expansions feed directly into the
  real `websearch_to_tsquery` syntax the retrieval layer already speaks
- Robust response parsing: strict JSON happy path, markdown-fence stripping,
  JSON-substring extraction from surrounding prose, and a line-based fallback
  if the model ignores the JSON instruction entirely
- `build_websearch_query`: bridges expansion output into a real lexical query

**Real finding from live testing (not a hypothetical)**: the first version
phrase-quoted every multi-word expansion, assuming the LLM would echo
document-like phrasing. Testing against the live local model showed the
opposite -- it paraphrases ("IP rights infringement" for a document that says
"patent infringement"), so exact-phrase and even AND-of-words matching missed
everything. Fixed by also decomposing multi-word expansions into their
individual significant words, OR'd in alongside the phrase. Verified end-to-end:
a query worded so it shares zero terminology with the document ("IP rights
breach dispute") returns zero literal lexical hits, but the same query run
through Synonimise + the decomposed expansion returns real hits from the
actual `document_chunks` content -- confirmed with the live local Ollama model,
not mocked.

## What's implemented (Phase 3)

- Dense retrieval via a Postgres RPC (`match_document_chunks_dense`, pgvector
  `<=>` cosine distance) and lexical/BM25-equivalent retrieval via another
  (`match_document_chunks_lexical`, Postgres full-text on the `content_tsv` column)
- Boolean/exact-phrase/exclusion search is **not** a bespoke parser: lexical
  search is driven directly by Postgres's built-in `websearch_to_tsquery`,
  which already supports quoted phrases, implicit AND, `OR`, and `-exclusion`
- Metadata filtering by project/KB-category/document, enforced inside the SQL
  functions (not just application code)
- Fusion via Reciprocal Rank Fusion (rank-based, not raw-score blending --
  dense cosine distance and lexical ts_rank live on incomparable scales)
- `SearchService` (`dense_search` / `lexical_search` / `hybrid_search`)
- Verified end-to-end against real Supabase + real BGE-M3: exact-phrase,
  boolean OR, exclusion (`-term`), semantic-only queries, hybrid fusion
  ordering, and cross-project scope isolation (a second project with no
  documents correctly returns zero results) all confirmed with real data,
  not mocked

## What's implemented (Phase 2)

- Docling-based PDF parsing that preserves headings, page numbers, and tables
  (OCR off by default for born-digital documents; pass `ocr=True` for scanned ones)
- Structure-aware chunking: sections stay whole when they fit the size budget,
  only oversized sections get sub-split (with overlap), every chunk keeps its
  heading/section/page
- Local BGE-M3 embedding provider (`sentence-transformers`, cached in-process)
  wired into the provider registry as the `embeddings` task
- `DocumentService.ingest_pdf`: uploads the original file to Supabase Storage,
  parses, chunks, embeds, and writes rows to `documents`/`document_chunks`;
  failures mark the document `error` and remove partial chunks rather than
  leaving it stuck `processing`
- Verified end-to-end against the real local Supabase + real BGE-M3 model:
  ingested a synthetic multi-page PDF (with a table), confirmed lexical
  full-text search and cosine-similarity vector search both return correct,
  topically relevant results — not mocked

**Finding worth knowing for later work**: PostgREST returns `vector` columns
as their text form (`"[0.1,0.2,...]"`), not a native array. Any Python code
that needs raw embedding values back from a query must parse that string —
better still, do similarity math in SQL via pgvector operators and avoid
pulling vectors into Python at all (this is what Phase 3's retrieval layer will do).

## What's implemented (Phase 1)

- Centralized config + 3 built-in provider profiles
- Provider interfaces (`LLMProvider`, `EmbeddingProvider`, `RerankerProvider`,
  `QueryExpansionProvider`) with OpenAI / Gemini / Ollama LLM implementations
- Privacy-mode enforcement with no silent cloud fallback
- Multi-tenant Postgres schema: organizations, membership, projects,
  knowledge-base categories, documents, chunks (pgvector + full-text), research
  sessions — with Row Level Security scoping every table to org membership
- 13 passing tests: unit tests (config/registry, no network calls) plus real
  integration smoke tests against a live local Supabase instance and a live
  local Ollama daemon

## Not yet implemented

- DOCX/TXT/HTML/other format ingestion (PDF only so far); OCR path untested
- Reranking
- A CLI (Streamlit UI exists; no CLI entry point yet)
- Password reset / email confirmation flows (local dev has confirmation
  disabled; a hosted deployment would need this before real launch)
- Async/background research execution and mid-run pause/resume from the UI
- Normalized findings/evidence tables (still intentionally kept as jsonb on
  `research_sessions` -- no cross-session querying need has appeared yet)
