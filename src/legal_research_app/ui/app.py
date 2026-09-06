"""Streamlit internal tool -- simplified per direction to focus on the core
engine rather than multi-tenancy. Single implicit workspace, no login: auth
(ui/auth.py) and its RPCs/migrations are left in the codebase unused, in case
real multi-user is wanted again later, but nothing in this file uses them.
Knowledge Base categories (a second collection type alongside Projects) are
similarly dropped from the UI -- Projects are now the only collection concept.

Primary flow: pick/create a Project (sidebar) -> upload document(s) -> ask a
question. Two research paths on the Research page:
  - Simple (default): one-pass semantic + keyword-expansion search -> one AI
    call that answers and cross-checks itself against the retrieved chunks.
    This is the core engine loop as described directly.
  - Advanced (QUICK/STANDARD/DEEP): the fuller iterative agent from Phase 5
    (contradiction detection, gap analysis, multi-round loop) -- kept
    available, not the default.

Provider: OpenAI only is exposed here (the multi-provider registry still
exists underneath and is reused as-is; profile switching just isn't surfaced
in this UI anymore).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from legal_research_app.agent.types import ResearchMode
from legal_research_app.config import DEFAULT_PROFILES, Settings
from legal_research_app.db.client import build_service_client
from legal_research_app.providers.registry import ProviderRegistry
from legal_research_app.retrieval.types import RetrievalScope
from legal_research_app.services.document_service import DocumentService
from legal_research_app.services.research_service import ResearchService
from legal_research_app.services.search_service import SearchService
from legal_research_app.ui import data

st.set_page_config(page_title="Foolproof AI", page_icon="🛡️", layout="wide")

ACTIVE_PROFILE = "hybrid_openai_default"
DEFAULT_ORG_NAME = "Default Workspace"

LOGO_SVG = """
<svg width="32" height="32" viewBox="0 0 36 36" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="fp-grad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#4f46e5"/>
      <stop offset="100%" stop-color="#9333ea"/>
    </linearGradient>
  </defs>
  <rect width="36" height="36" rx="9" fill="url(#fp-grad)"/>
  <path d="M11 9h14l-3 4H14v4h9l-3 4h-6v6h-3V9z" fill="white"/>
</svg>
"""


def render_brand() -> None:
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
        {LOGO_SVG}<div><div style="font-weight:700;font-size:1.15rem;line-height:1.1;">Foolproof AI</div>
        <div style="font-size:0.75rem;color:#8b8b8b;">Research deeper. Miss less.</div></div></div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------

@st.cache_resource
def get_db():
    return build_service_client()


@st.cache_resource
def get_registry() -> ProviderRegistry:
    return ProviderRegistry(Settings(active_profile=ACTIVE_PROFILE))


@st.cache_resource
def get_default_org_id() -> str:
    """Single implicit workspace -- no login, no org picker. Uses the
    service-role client directly (bypassing RLS), same as before auth
    existed; RLS/multi-tenancy stay dormant in the schema, not deleted."""
    db = get_db()
    existing = db.table("organizations").select("id").order("created_at").limit(1).execute().data
    if existing:
        return existing[0]["id"]
    org = (
        db.table("organizations")
        .insert({"name": DEFAULT_ORG_NAME, "privacy_mode": "hybrid", "active_provider_profile": ACTIVE_PROFILE})
        .execute()
        .data[0]
    )
    return org["id"]


PRIVACY_BANNER = {
    "local_only": ("🔒 LOCAL ONLY", "No document content or question text leaves this machine/server.", "success"),
    "hybrid": ("🔓 HYBRID", "Documents and embeddings stay local. Evidence excerpts are sent to OpenAI for answers.", "warning"),
    "cloud_assisted": ("☁️ CLOUD ASSISTED", "Broader cloud model usage is enabled for this profile.", "error"),
}


def render_privacy_banner(registry: ProviderRegistry, *, compact: bool = False) -> None:
    label, detail, kind = PRIVACY_BANNER[registry.privacy_mode.value]
    text = label if compact else f"**{label}** — {detail}"
    getattr(st, kind)(text)


def render_provider_health(registry: ProviderRegistry) -> None:
    for name, health in registry.health_report().items():
        icon = "🟢" if health.available else "🔴"
        st.caption(f"{icon} {name}: {health.detail}")


# --------------------------------------------------------------------------
# Sidebar: brand, project picker/creator/rename
# --------------------------------------------------------------------------

db = get_db()
registry = get_registry()
org_id = get_default_org_id()

with st.sidebar:
    render_brand()
    st.divider()
    st.caption("PROJECT")

    projects = data.list_projects(db, org_id)
    st.session_state.setdefault("project_id", None)
    if st.session_state["project_id"] is None and projects:
        st.session_state["project_id"] = projects[0]["id"]

    if projects:
        try:
            default_index = next(i for i, p in enumerate(projects) if p["id"] == st.session_state["project_id"])
        except StopIteration:
            default_index = 0
        selected_project = st.selectbox(
            "Project", projects, format_func=lambda p: p["name"], index=default_index, label_visibility="collapsed"
        )
        st.session_state["project_id"] = selected_project["id"]
    else:
        st.caption("No projects yet.")

    if st.button("+ New Project", use_container_width=True):
        new_project = data.create_project(db, org_id, f"Project {len(projects) + 1}", None, None)
        st.session_state["project_id"] = new_project["id"]
        st.rerun()

    project_id = st.session_state["project_id"]
    current_project = next((p for p in projects if p["id"] == project_id), None)
    if current_project:
        with st.popover("Rename project", use_container_width=True):
            new_name = st.text_input("Name", value=current_project["name"], key=f"rename_{project_id}")
            if st.button("Save", key=f"save_rename_{project_id}"):
                db.table("projects").update({"name": new_name}).eq("id", project_id).execute()
                st.rerun()

    st.divider()
    st.caption("SYSTEM STATUS")
    render_provider_health(registry)
    render_privacy_banner(registry, compact=True)

if not project_id:
    st.title("Welcome to Foolproof AI")
    st.info("Click **+ New Project** in the sidebar to get started.")
    st.stop()


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

def page_research() -> None:
    st.title("Research")
    st.caption(f"Project: {current_project['name']}")

    question = st.text_area("Ask a question about your documents", height=100)

    mode_choice = st.radio("Mode", ["Simple", "Advanced", "Verified Research"], horizontal=True)
    research_mode = ResearchMode.SIMPLE
    jurisdiction = None
    if mode_choice == "Advanced":
        research_mode = st.selectbox(
            "Research depth",
            [ResearchMode.QUICK, ResearchMode.STANDARD, ResearchMode.DEEP],
            format_func=lambda m: m.value.upper(),
        )
        jurisdiction = st.text_input("Jurisdiction (optional)") or None
    elif mode_choice == "Verified Research":
        research_mode = ResearchMode.VERIFIED
        jurisdiction = st.text_input("Jurisdiction (optional)") or None
        tavily_health = registry.web_search().health() if registry.privacy_mode.value != "local_only" else None
        if tavily_health and not tavily_health.available:
            st.warning(f"Web search unavailable: {tavily_health.detail}")
        st.caption(
            "Local documents + web search restricted to primary Indian legal sources "
            "(indiankanoon.org, sci.gov.in, gov.in/nic.in) plus established legal commentary. "
            "Primary sources are weighted over secondary ones when they conflict. Slower and "
            "uses external web calls -- use for questions that need verification beyond your documents."
        )
    else:
        st.caption("Semantic search + keyword expansion + keyword search → one AI answer, cross-checked against the retrieved evidence.")

    if st.button("Ask", type="primary", disabled=not question.strip()):
        service = ResearchService(db, registry)
        if research_mode == ResearchMode.SIMPLE:
            with st.spinner("Searching and cross-checking..."):
                session_id = service.run_simple_query(org_id=org_id, question=question, project_id=project_id)
        else:
            with st.spinner(f"Researching in {research_mode.value.upper()} mode — this may take a while..."):
                session_id = service.start_research(
                    org_id=org_id,
                    question=question,
                    project_id=project_id,
                    jurisdiction=jurisdiction,
                    research_mode=research_mode,
                )
        st.session_state["last_session_id"] = session_id
        st.rerun()

    if "last_session_id" in st.session_state:
        st.divider()
        session = data.get_research_session(db, st.session_state["last_session_id"])

        status_kind = {"completed": "success", "failed": "error", "paused": "warning"}.get(session["status"], "info")
        getattr(st, status_kind)(f"Status: {session['status']} (mode: {session['research_mode']})")

        with st.expander("Operational trace", expanded=session["status"] != "completed"):
            for event in session["operational_trace"]:
                st.write(f"→ **{event['step']}**: {event['detail']}")

        answer = session.get("final_answer")
        if answer:
            st.subheader("Answer")
            st.write(answer["executive_answer"])

            conf = answer["confidence"]
            conf_kind = {"high": "success", "medium": "warning", "low": "error"}[conf["level"]]
            getattr(st, conf_kind)(f"Confidence: {conf['level'].upper()}")
            for line in conf["rationale"]:
                st.caption(f"• {line}")

            evidence_by_index = {e["index"]: e for e in answer["evidence"]}

            if answer["key_findings"]:
                st.subheader("Key Findings")
                for finding in answer["key_findings"]:
                    with st.expander(finding["statement"]):
                        for idx in finding["evidence_indices"]:
                            ev = evidence_by_index.get(idx)
                            if ev:
                                page = f", p.{ev['page_number']}" if ev["page_number"] else ""
                                st.caption(f"[{idx}] {ev['document_filename']}{page}: {ev['content'][:300]}")

            if answer["contradictions"]:
                st.subheader("Contradictions")
                for c in answer["contradictions"]:
                    st.warning(c["explanation"])

            if answer["gaps"]:
                st.subheader("Gaps / Limitations")
                for g in answer["gaps"]:
                    st.caption(f"• {g}")

            st.subheader("Research Coverage")
            for line in answer["research_coverage"]:
                st.caption(f"• {line}")

            st.subheader("Evidence")
            authority_badge = {"user_document": "📄 your document", "primary": "🏛️ primary source", "secondary": "📰 secondary source"}
            for ev in answer["evidence"]:
                page = f", p.{ev['page_number']}" if ev["page_number"] else ""
                heading = f" — {ev['heading']}" if ev["heading"] else ""
                badge = authority_badge.get(ev.get("source_authority", "user_document"), "")
                with st.expander(f"[{ev['index']}] {badge} {ev['document_filename']}{page}{heading}"):
                    st.write(ev["content"])
                    if ev.get("source_url"):
                        st.caption(f"[{ev['source_url']}]({ev['source_url']})")
                    st.caption(f"matched by: {', '.join(ev['matched_by'])} — score {ev['score']:.3f}")


def page_documents() -> None:
    st.title("Documents")
    st.caption(f"Project: {current_project['name']}")
    st.caption("PDF only for now — DOCX/TXT/HTML are not implemented yet.")

    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded_files and st.button("Ingest all"):
        service = DocumentService(db, registry)
        for uploaded in uploaded_files:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir) / uploaded.name
                tmp_path.write_bytes(uploaded.getvalue())
                with st.spinner(f"Parsing, chunking, and embedding {uploaded.name}..."):
                    result = service.ingest_pdf(
                        org_id=org_id, file_path=tmp_path, filename=uploaded.name, project_id=project_id
                    )
            if result.status == "indexed":
                st.success(f"{uploaded.name}: indexed, {result.chunk_count} chunks, {result.page_count} page(s).")
            else:
                st.error(f"{uploaded.name}: {result.error_message}")

    st.divider()
    st.subheader("Document library")
    documents = data.list_documents(db, org_id, project_id=project_id)
    if not documents:
        st.caption("No documents yet.")
    else:
        for doc in documents:
            status_icon = {"indexed": "✅", "processing": "⏳", "pending": "⏳", "error": "❌"}.get(doc["status"], "❓")
            line = f"{status_icon} **{doc['filename']}** — {doc['status']}"
            if doc.get("page_count"):
                line += f" — {doc['page_count']} page(s)"
            st.write(line)
            if doc.get("error_message"):
                st.caption(f"⚠️ {doc['error_message']}")


def page_search_builder() -> None:
    """Manual, low-level access to SearchService -- useful for debugging
    retrieval quality directly, without going through the full research agent."""
    st.title("Search Builder")
    st.caption(f"Project: {current_project['name']} — run dense, lexical, or fused search directly against this project's index.")

    search_kind = st.selectbox("Search type", ["Hybrid (fused)", "Dense (semantic)", "Lexical (keyword/boolean)"])
    query = st.text_input(
        "Query",
        placeholder='e.g. "patent infringement" or infringement OR violation -settlement',
    )
    match_count = st.slider("Max results", 1, 30, 10)

    suggest_col, search_col = st.columns([1, 1])
    if suggest_col.button("💡 Suggest terms from corpus", disabled=not query.strip()):
        scope = RetrievalScope(org_id=org_id, project_ids=[project_id])
        search = SearchService(db, registry)
        with st.spinner("Finding recurring corpus terms near your query..."):
            st.session_state["suggested_terms"] = search.suggest_terms(query, scope)

    if st.session_state.get("suggested_terms") is not None:
        terms = st.session_state["suggested_terms"]
        with st.expander(f"Suggested terms ({len(terms)}) — real phrasing pulled from your documents, not AI-generated", expanded=True):
            if not terms:
                st.caption("No recurring phrases found across the nearest-neighbor chunks for this query.")
            for t in terms:
                st.caption(f"**{t.text}** — appears in {t.chunk_count} matched chunks")

    if search_col.button("Search", type="primary", disabled=not query.strip()):
        scope = RetrievalScope(org_id=org_id, project_ids=[project_id])
        search = SearchService(db, registry)
        with st.spinner("Searching..."):
            if search_kind.startswith("Dense"):
                hits = search.dense_search(query, scope, match_count=match_count)
            elif search_kind.startswith("Lexical"):
                hits = search.lexical_search(query, scope, match_count=match_count)
            else:
                hits = search.hybrid_search(query, scope, match_count=match_count)

        st.caption(f"{len(hits)} result(s)")
        for hit in hits:
            with st.expander(f"score {hit.score:.3f} — {', '.join(hit.matched_by)} — {hit.heading or '(no heading)'}"):
                if hit.page_number:
                    st.caption(f"page {hit.page_number}")
                st.write(hit.content)


def page_history() -> None:
    st.title("Research History")
    st.caption(f"Project: {current_project['name']}")
    sessions = data.list_research_sessions(db, org_id, project_id=project_id)
    if not sessions:
        st.caption("No research sessions yet.")
    for s in sessions:
        cols = st.columns([5, 1, 1, 2])
        cols[0].write(s["question"])
        cols[1].write(s["research_mode"])
        cols[2].write(s["status"])
        if cols[3].button("Open in Research", key=f"open_{s['id']}"):
            st.session_state["last_session_id"] = s["id"]
            st.switch_page(research_page)


def page_settings() -> None:
    st.title("Settings")
    st.subheader("AI Provider")
    render_privacy_banner(registry)
    render_provider_health(registry)
    st.caption("This app is configured to use OpenAI. Update OPENAI_API_KEY in .env to change credentials.")

    st.subheader("Model assignments")
    profile = DEFAULT_PROFILES[ACTIVE_PROFILE]
    for task, cfg in profile.tasks.items():
        st.caption(f"{task.value} → {cfg.provider.value} / {cfg.model}")

    st.subheader("Verified Research (web search)")
    try:
        tavily_health = registry.web_search().health()
        icon = "🟢" if tavily_health.available else "🔴"
        st.caption(f"{icon} tavily: {tavily_health.detail}")
    except Exception as exc:
        st.caption(f"🔴 tavily: {exc}")
    st.caption("Get a free key at tavily.com and set TAVILY_API_KEY in .env to enable Verified Research mode.")


# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------

research_page = st.Page(page_research, title="Research", icon="🔍", default=True)
documents_page = st.Page(page_documents, title="Documents", icon="📄")
search_builder_page = st.Page(page_search_builder, title="Search Builder", icon="🧪")
history_page = st.Page(page_history, title="Research History", icon="🕘")
settings_page = st.Page(page_settings, title="Settings", icon="⚙️")

pg = st.navigation([research_page, documents_page, search_builder_page, history_page, settings_page])
pg.run()
