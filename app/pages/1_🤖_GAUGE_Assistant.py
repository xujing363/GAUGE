import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

import streamlit as st  # noqa: E402

from gauge_core.agent import AgentNotConfiguredError, GaugeAgent, ToolCallRecord  # noqa: E402
from gauge_core.expression_io import parse_expression_table  # noqa: E402
from gauge_core import providers  # noqa: E402

import json as _json  # noqa: E402
from datetime import datetime  # noqa: E402


# ── Conversation transcript (de)serialisation ─────────────────────────────────
def _history_to_markdown(history: list) -> str:
    """Render the chat transcript (with tool calls) as a saveable Markdown document."""
    lines = [f"# GAUGE Assistant conversation\n\n_Saved {datetime.now():%Y-%m-%d %H:%M}_\n"]
    for m in history:
        who = "🧑 **User**" if m["role"] == "user" else "🤖 **GAUGE Assistant**"
        lines.append(f"\n## {who}\n\n{m['content']}\n")
        for tc in m.get("tool_calls", []):
            lines.append(f"- 🔧 `{tc.name}` — args: `{tc.args}`")
    return "\n".join(lines)


def _history_to_json(history: list) -> str:
    """Serialise the transcript (incl. tool calls) to JSON so it can be reloaded."""
    payload = [
        {
            "role": m["role"],
            "content": m["content"],
            "tool_calls": [{"name": tc.name, "args": tc.args, "result": tc.result} for tc in m.get("tool_calls", [])],
        }
        for m in history
    ]
    return _json.dumps({"version": 1, "messages": payload}, default=str, ensure_ascii=False, indent=2)


def _history_from_json(raw: bytes) -> list:
    """Reconstruct session history (with ToolCallRecord objects) from saved JSON."""
    data = _json.loads(raw)
    messages = data["messages"] if isinstance(data, dict) else data
    restored = []
    for m in messages:
        restored.append(
            {
                "role": m["role"],
                "content": m["content"],
                "tool_calls": [
                    ToolCallRecord(name=tc["name"], args=tc.get("args", {}), result=tc.get("result", {}))
                    for tc in m.get("tool_calls", [])
                ],
            }
        )
    return restored


# ── Multi-conversation session store ──────────────────────────────────────────
def _new_conv(title: str = "New chat", history: list | None = None) -> dict:
    return {"id": uuid.uuid4().hex[:8], "title": title, "history": history or []}


def _ensure_conversations() -> None:
    """One-time migration: lift any legacy single-history session into the
    conversation list, then guarantee an active conversation exists."""
    if "conversations" not in st.session_state:
        legacy = st.session_state.get("assistant_history")
        if legacy:
            conv = _new_conv(_title_from_history(legacy), legacy)
        else:
            conv = _new_conv()
        st.session_state["conversations"] = [conv]
        st.session_state["active_conv_id"] = conv["id"]


def _active_conv() -> dict:
    convs = st.session_state["conversations"]
    cid = st.session_state.get("active_conv_id")
    for c in convs:
        if c["id"] == cid:
            return c
    st.session_state["active_conv_id"] = convs[0]["id"]
    return convs[0]


def _title_from_history(history: list) -> str:
    first_user = next((m["content"] for m in history if m["role"] == "user"), "")
    title = " ".join(first_user.split())[:42]
    return (title + "…") if len(first_user) > 42 else (title or "New chat")


common.configure_page("GAUGE Assistant", icon="🤖")
_ensure_conversations()
# NB: the Model / provider / mode sidebar block is rendered first (below, after the
# CSS) so it appears ABOVE the GAUGE checkpoint selector; the GAUGE bundle selector
# is rendered afterwards via common.sidebar_mode_selector().

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Page background */
    .block-container { padding-top: 1.5rem; }

    /* Welcome hero */
    .gauge-hero {
        background: linear-gradient(135deg, #1a3a5c 0%, #2E86AB 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    .gauge-hero h1 { color: white; font-size: 2rem; margin: 0 0 0.3rem 0; }
    .gauge-hero p  { color: rgba(255,255,255,0.85); margin: 0; font-size: 1.05rem; }

    /* Feature cards */
    .feature-grid { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
    .feature-card {
        flex: 1; min-width: 160px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.1rem;
    }
    .feature-card .icon { font-size: 1.5rem; margin-bottom: 0.4rem; }
    .feature-card .title { font-weight: 600; font-size: 0.88rem; color: #1e293b; margin-bottom: 0.2rem; }
    .feature-card .desc  { font-size: 0.78rem; color: #64748b; line-height: 1.4; }

    /* Quick-start cards */
    .stButton > button {
        width: 100%;
        white-space: normal !important;
        text-align: left !important;
        height: auto !important;
        padding: 0.7rem 0.9rem !important;
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
        background: #f8fafc !important;
        color: #1e293b !important;
        font-size: 0.82rem !important;
        line-height: 1.35 !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        border-color: #2E86AB !important;
        background: #eff8ff !important;
        color: #1a3a5c !important;
    }

    /* API key status badge */
    .key-badge-ok  { background:#dcfce7; color:#166534; padding:3px 10px; border-radius:99px; font-size:0.78rem; display:inline-block; }
    .key-badge-no  { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:99px; font-size:0.78rem; display:inline-block; }

    /* Disclaimer banner */
    .disclaimer {
        background: #fff7ed;
        border-left: 4px solid #f97316;
        border-radius: 0 8px 8px 0;
        padding: 0.5rem 1rem;
        font-size: 0.83rem;
        color: #7c2d12;
        margin-bottom: 1rem;
    }

    /* Tool call expanders — subtle */
    .streamlit-expanderHeader { font-size: 0.8rem !important; color: #64748b !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar: conversations (kept on the left, ChatGPT-style) ──────────────────
st.sidebar.markdown("### 💬 Conversations")
if st.sidebar.button("➕ New chat", key="new_chat", use_container_width=True):
    active = _active_conv()
    # Reuse the current chat if it's still empty rather than piling up blanks.
    if active["history"]:
        conv = _new_conv()
        st.session_state["conversations"].insert(0, conv)
        st.session_state["active_conv_id"] = conv["id"]
    st.rerun()

for _c in st.session_state["conversations"]:
    _is_active = _c["id"] == st.session_state["active_conv_id"]
    _label = ("🟢 " if _is_active else "💬 ") + (_c["title"] or "New chat")
    if st.sidebar.button(_label, key=f"conv_{_c['id']}", use_container_width=True):
        st.session_state["active_conv_id"] = _c["id"]
        st.rerun()

conv = _active_conv()
with st.sidebar.expander("✏️ Rename / delete current chat", expanded=False):
    _new_title = st.text_input("Title", value=conv["title"], key=f"title_{conv['id']}")
    if _new_title and _new_title != conv["title"]:
        conv["title"] = _new_title
    if st.button("🗑️ Delete this chat", key=f"del_{conv['id']}", use_container_width=True):
        st.session_state["conversations"] = [c for c in st.session_state["conversations"] if c["id"] != conv["id"]]
        if not st.session_state["conversations"]:
            st.session_state["conversations"] = [_new_conv()]
        st.session_state["active_conv_id"] = st.session_state["conversations"][0]["id"]
        st.rerun()

st.sidebar.divider()

# ── Sidebar: model provider + API key (user supplies their own) ───────────────
st.sidebar.markdown("### 🧠 Model")
# Only tool-capable models can drive the assistant (it relies on function calling).
# Reasoning models (e.g. deepseek-reasoner) are used internally for Deep Report
# synthesis only, so they are intentionally excluded from this chat-model selector.
_PROVIDER_NAMES = [n for n, p in providers.PROVIDERS.items() if p.supports_tools]
_default_idx = _PROVIDER_NAMES.index(providers.DEFAULT_PROVIDER) if providers.DEFAULT_PROVIDER in _PROVIDER_NAMES else 0
provider_name = st.sidebar.selectbox(
    "Chat model",
    _PROVIDER_NAMES,
    index=_default_idx,
    format_func=lambda n: providers.PROVIDERS[n].label,
    key="assistant_provider",
)
selected_provider = providers.PROVIDERS[provider_name]

st.sidebar.markdown("### 🔑 Your API Key")
st.sidebar.caption(
    f"Powered by **{selected_provider.label}**. Enter **your own key** below — it is never "
    "stored and is only used within your browser session."
)
api_key = st.sidebar.text_input(
    "API key",
    type="password",
    placeholder="sk-...",
    key=f"assistant_api_key_{provider_name}",
    label_visibility="collapsed",
)
if api_key:
    st.sidebar.markdown('<span class="key-badge-ok">✓ Key configured</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="key-badge-no">No key entered</span>', unsafe_allow_html=True)
    if provider_name.startswith("deepseek"):
        st.sidebar.markdown(
            "Get a free key at [platform.deepseek.com](https://platform.deepseek.com) (requires registration).",
        )

# ── Sidebar: response mode + process visibility + external databases ──────────
st.sidebar.markdown("### 🧪 Assistant mode")
mode_choice = st.sidebar.radio(
    "Response style",
    ["⚡ Quick Analysis", "📋 Deep Report"],
    key="assistant_mode",
    help="Quick Analysis answers conversationally. Deep Report plans, gathers evidence across "
    "many tool calls, and writes a structured, cited report (slower; uses a reasoning model).",
    label_visibility="collapsed",
)
is_report_mode = mode_choice.startswith("📋")

show_process = st.sidebar.toggle(
    "👁️ Show process live",
    value=True,
    key="assistant_show_process",
    help="Show the assistant's plan and each tool call as it works. In Deep Report mode this "
    "reveals the full plan → gather evidence → write → self-review pipeline. Turn off for a "
    "cleaner view that only shows the final answer.",
)

enable_external = st.sidebar.toggle(
    "🌐 External biomedical databases",
    value=False,
    key="assistant_external",
    help="Let the assistant query OpenTargets, PubChem, UniProt, ChEMBL, DGIdb, Reactome, "
    "cBioPortal, ClinicalTrials.gov and the literature (Europe PMC) for target biology, drug "
    "pharmacology, mutation frequency, trials and citations. Many of these answer drug-sensitivity "
    "questions without needing a GAUGE prediction. Requires internet access; GAUGE predictions are "
    "unaffected when this is off.",
)

# ── Sidebar: save / load the conversation ─────────────────────────────────────
with st.sidebar.expander("💾 Save / load conversation", expanded=False):
    _hist = conv["history"]
    _stamp = datetime.now().strftime("%Y%m%d_%H%M")
    if _hist:
        cdl1, cdl2 = st.columns(2)
        cdl1.download_button(
            "⬇️ JSON", _history_to_json(_hist), file_name=f"gauge_chat_{_stamp}.json",
            mime="application/json", key="save_chat_json", use_container_width=True,
        )
        cdl2.download_button(
            "⬇️ Markdown", _history_to_markdown(_hist), file_name=f"gauge_chat_{_stamp}.md",
            mime="text/markdown", key="save_chat_md", use_container_width=True,
        )
    else:
        st.caption("Start chatting, then come back here to save the conversation.")
    _loaded = st.file_uploader("Load a saved .json conversation", type=["json"], key="load_chat_file")
    if _loaded is not None and st.session_state.get("_loaded_chat_id") != _loaded.file_id:
        try:
            restored = _history_from_json(_loaded.getvalue())
            new_conv = _new_conv(_title_from_history(restored), restored)
            st.session_state["conversations"].insert(0, new_conv)
            st.session_state["active_conv_id"] = new_conv["id"]
            st.session_state["_loaded_chat_id"] = _loaded.file_id
            st.success(f"Loaded {len(restored)} messages into a new chat.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load conversation: {exc}")

st.sidebar.divider()

# ── GAUGE checkpoint selector (rendered below the Model controls) ──────────────
bundle = common.sidebar_mode_selector()

# ── Sidebar: expression file upload ──────────────────────────────────────────
with st.sidebar.expander("📂 Upload expression file", expanded=False):
    st.caption(
        "Optional. Upload a CSV/TSV and refer to sample names directly in the chat "
        "(e.g. *\"Rank drugs for sample ACH-000007\"*)."
    )
    uploaded_file = st.file_uploader(
        "Expression file", type=["csv", "tsv", "txt"], key="assistant_upload",
        label_visibility="collapsed",
    )
    gdsc_demo_path = common.EXAMPLE_DATA_DIR / "example_expression_multi_sample.csv"
    prism_demo_path = common.EXAMPLE_DATA_DIR / "example_expression_prism_demo.csv"
    c1, c2 = st.columns(2)
    if gdsc_demo_path.exists():
        c1.download_button("⬇️ GDSC template", gdsc_demo_path.read_bytes(),
                           file_name="gauge_expr_gdsc.csv", mime="text/csv", key="asst_dl_gdsc")
    if prism_demo_path.exists():
        c2.download_button("⬇️ PRISM template", prism_demo_path.read_bytes(),
                           file_name="gauge_expr_prism.csv", mime="text/csv", key="asst_dl_prism")

uploaded_samples: dict = {}
if uploaded_file is not None:
    try:
        uploaded_samples = parse_expression_table(uploaded_file, bundle.artifacts.genes)
        names = list(uploaded_samples.keys())
        st.sidebar.success(
            f"Loaded **{len(names)}** sample{'s' if len(names) != 1 else ''}: "
            + ", ".join(f"`{s}`" for s in names[:4])
            + (" …" if len(names) > 4 else "")
        )
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Could not parse file: {exc}")

_PRISM_MODES = getattr(common, "PRISM_MODES", {"prism_cell_split", "prism_drug_split"})
_is_prism = bundle.mode in _PRISM_MODES

history = conv["history"]


# ── Live-process helpers ──────────────────────────────────────────────────────
def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])


def _make_progress(status):
    """Return a callback the agent calls at each research stage; it writes the
    plan and live tool calls into the given st.status container."""
    _STAGE_LABEL = {
        "plan": "🧭 Planning the research…",
        "gather": "🔎 Gathering evidence with tools…",
        "synthesis": "✍️ Writing the report…",
        "critique": "🔍 Reviewing the draft for unsupported claims…",
        "done": "✅ Done",
    }

    def _cb(stage: str, info: dict) -> None:
        if stage == "plan" and info.get("final"):
            if info.get("text"):
                status.markdown("**Research plan**\n\n" + info["text"])
            status.update(label="🔎 Gathering evidence with tools…")
        elif stage == "tool":
            rec = info["record"]
            err = isinstance(rec.result, dict) and rec.result.get("error")
            icon = "⚠️" if err else "🔧"
            status.markdown(f"{icon} `{rec.name}` — {_fmt_args(rec.args)}")
        elif stage in _STAGE_LABEL:
            status.update(label=_STAGE_LABEL[stage], state="complete" if stage == "done" else "running")

    return _cb


# ── Welcome / empty state ─────────────────────────────────────────────────────
clicked_demo = None
if not history:
    st.markdown(
        """
        <div class="gauge-hero">
          <h1>🤖 GAUGE Assistant</h1>
          <p>A virtual-biologist AI agent that plans, calls GAUGE in real time, and pulls
          target biology, drug pharmacology and literature to reason about drug response —
          every GAUGE number comes from the model, never invented by the LLM.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-grid">
          <div class="feature-card">
            <div class="icon">🔬</div>
            <div class="title">Predict &amp; rank</div>
            <div class="desc">Ask how sensitive a sample is to a drug, or rank the whole library for it.</div>
          </div>
          <div class="feature-card">
            <div class="icon">🧠</div>
            <div class="title">Why-it-predicted</div>
            <div class="desc">Explain a prediction via knowledge-graph source-attention and the drug's KG neighbourhood.</div>
          </div>
          <div class="feature-card">
            <div class="icon">💊</div>
            <div class="title">Drug sensitivity (no GAUGE needed)</div>
            <div class="desc">Druggable targets (DGIdb), mechanism &amp; phase (ChEMBL), trials, pathways, mutation frequency.</div>
          </div>
          <div class="feature-card">
            <div class="icon">🌐</div>
            <div class="title">Target biology</div>
            <div class="desc">OpenTargets, UniProt, Reactome, cBioPortal and PubChem for target evidence and citations.</div>
          </div>
          <div class="feature-card">
            <div class="icon">📋</div>
            <div class="title">Deep reports</div>
            <div class="desc">Plan → gather evidence → write a structured, cited report — with the process shown live.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Try one of these examples to get started:**")
    _sample = "ACH-000007" if _is_prism else "SIDM00003"
    examples = [
        ("🔬 Single prediction", f"What does GAUGE predict for {_sample} treated with Erlotinib?"),
        ("🧠 Explain why", "Why did GAUGE predict that? Which knowledge graph drove it?"),
        ("💊 Drug sensitivity (no GAUGE)", "What drugs target EGFR, and what is Erlotinib's mechanism and clinical phase?"),
        ("🌐 Target biology", "Is EGFR a good target in lung adenocarcinoma? How often is it mutated, and cite the evidence."),
        ("📋 Deep report", f"Write a report on Erlotinib for {_sample}: GAUGE prediction, EGFR biology, trials and literature."),
    ]

    cols = st.columns(2)
    for i, (label, q) in enumerate(examples):
        col = cols[i % 2]
        if col.button(f"{label}\n\n_{q}_", key=f"demo_q_{i}"):
            clicked_demo = q

    st.markdown(
        '<div class="disclaimer">⚠️ <strong>Research use only.</strong> '
        "GAUGE predictions are computational hypotheses for drug-response research, "
        "not clinical diagnoses or treatment recommendations.</div>",
        unsafe_allow_html=True,
    )

else:
    # ── Compact header when chat is active ───────────────────────────────────
    c_title, c_save, c_clear = st.columns([4, 1, 1])
    c_title.markdown(f"### 🤖 {conv['title']}")
    c_save.download_button(
        "💾 Save",
        _history_to_markdown(history),
        file_name=f"gauge_chat_{datetime.now():%Y%m%d_%H%M}.md",
        mime="text/markdown",
        key="save_chat_header",
        use_container_width=True,
    )
    if c_clear.button("🗑️ Clear", key="clear_chat", use_container_width=True):
        conv["history"] = []
        st.rerun()

    # ── Chat history ─────────────────────────────────────────────────────────
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for tc in msg.get("tool_calls", []):
                label = f"🔧 `{tc.name}` — {_fmt_args(tc.args)}"
                with st.expander(label, expanded=False):
                    st.json(tc.result)

# ── Chat input ────────────────────────────────────────────────────────────────
if not api_key:
    st.info(
        f"👆 Enter your {selected_provider.label} API key in the sidebar to start chatting.",
        icon="🔑",
    )
else:
    placeholder = (
        "Ask for a deep, cited report on a target or drug…"
        if is_report_mode
        else "Ask about a prediction, a drug ranking, target biology, or drug sensitivity…"
    )
    user_input = st.chat_input(placeholder) or clicked_demo

    if user_input:
        history.append({"role": "user", "content": user_input})
        if conv["title"] in ("", "New chat"):
            conv["title"] = _title_from_history(history)
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            agent = GaugeAgent(
                bundle,
                api_key=api_key or None,
                provider=provider_name,
                uploaded_samples=uploaded_samples or None,
                enable_external=enable_external,
            )
        except AgentNotConfiguredError as exc:
            with st.chat_message("assistant"):
                st.error(str(exc))
        else:
            with st.chat_message("assistant"):
                result = None
                error_text = None
                # When the user wants the process shown, drive a live st.status
                # container via the agent's progress callback; otherwise fall back
                # to a plain spinner that only reveals the final answer.
                status = None
                progress = None
                if show_process:
                    status = st.status(
                        "🧭 Deep Report: starting…" if is_report_mode else "🔧 Working…",
                        expanded=True,
                    )
                    progress = _make_progress(status)

                def _run():
                    snapshot = [{"role": m["role"], "content": m["content"]} for m in history]
                    return (
                        agent.run_report(snapshot, progress=progress)
                        if is_report_mode
                        else agent.run_turn(snapshot, progress=progress)
                    )

                try:
                    if status is not None:
                        result = _run()
                        status.update(label="✅ Done", state="complete", expanded=False)
                    else:
                        spinner_text = (
                            "Researching: planning → gathering evidence → writing report…"
                            if is_report_mode
                            else "Calling GAUGE tools…"
                        )
                        with st.spinner(spinner_text):
                            result = _run()
                except Exception as exc:  # noqa: BLE001
                    hint = (
                        " — Deep Report uses a reasoning model and many tool calls, so it is slower and "
                        "more sensitive to API limits/timeouts. Try ⚡ Quick Analysis, or check that your "
                        "key has access to the selected model."
                        if is_report_mode
                        else ""
                    )
                    error_text = f"⚠️ LLM API error: {exc}{hint}"
                    if status is not None:
                        status.update(label="⚠️ Error", state="error")

                if error_text is not None:
                    st.error(error_text)
                    # Persist the error in the transcript so it doesn't vanish on rerun.
                    history.append({"role": "assistant", "content": error_text, "tool_calls": []})
                else:
                    st.markdown(result.reply)
                    if is_report_mode and result.reply:
                        st.download_button(
                            "⬇️ Download report (.md)",
                            result.reply,
                            file_name=f"gauge_report_{datetime.now():%Y%m%d_%H%M}.md",
                            mime="text/markdown",
                            key=f"dl_report_{len(history)}",
                        )
                    if result.tool_calls:
                        st.caption(f"🔎 Evidence trail — {len(result.tool_calls)} tool call(s)")
                    for tc in result.tool_calls:
                        label = f"🔧 `{tc.name}` — {_fmt_args(tc.args)}"
                        with st.expander(label, expanded=False):
                            st.json(tc.result)
                    history.append(
                        {"role": "assistant", "content": result.reply, "tool_calls": result.tool_calls}
                    )
        st.rerun()
