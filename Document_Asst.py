
import re
import uuid
import tempfile
from pathlib import Path

import streamlit as st

from config import SUPPORTED_EXTENSIONS, MAX_HISTORY_TURNS, MAX_IMAGES_PER_QUERY
from extractor import extract_document, cleanup_session_images
from ollama_client import (
    build_prompt,
    check_ollama_connection,
    stream_response,
    GenerationStats,
)
from logger import get_logger, activity_log

log = get_logger("app")


# ── page config ───────────────────────────────────────────────

st.set_page_config(
    page_title = "AI Document Assistant",
    page_icon  = "🤖",
    layout     = "wide",
)


# ── Ollama check — terminal/log only, NOT shown in UI ─────────

@st.cache_data(ttl=60, show_spinner=False)
def _cached_ollama_check() -> tuple:
    return check_ollama_connection()

_ok, _info = _cached_ollama_check()
log.info(f"Ollama | {'Connected' if _ok else 'NOT reachable'} | {_info}")


# ── client IP detection ───────────────────────────────────────

def _get_client_ip() -> str:
    """
    Detect client IP. Tries 4 approaches across Streamlit versions.
    Never crashes the app — returns 'unknown' on all failures.
    """
    # Approach 1: Streamlit 1.37+ (st.context.headers)
    try:
        headers = getattr(st.context, "headers", None)
        if headers:
            for key in ("x-forwarded-for", "x-real-ip", "remote-addr"):
                val = headers.get(key, "").split(",")[0].strip()
                if val and val != "unknown":
                    return val
    except Exception:
        pass

    # Approach 2: Streamlit runtime session info
    try:
        from streamlit.runtime import get_instance
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        runtime = get_instance()
        ctx     = get_script_run_ctx()
        if runtime and ctx:
            session_info = runtime.get_session_info(ctx.session_id)
            if session_info:
                request = getattr(session_info, "request", None)
                if request:
                    ip = getattr(request, "remote_ip", None) or \
                         getattr(request, "remote_addr", None)
                    if ip and ip != "unknown":
                        return str(ip)
    except Exception:
        pass

    # Approach 3: Script run context request headers
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx:
            req = getattr(ctx, "request", None)
            if req:
                hdrs = getattr(req, "headers", {})
                for key in ("x-forwarded-for", "x-real-ip"):
                    val = hdrs.get(key, "").split(",")[0].strip()
                    if val and val != "unknown":
                        return val
                ip = getattr(req, "remote_ip", None)
                if ip:
                    return str(ip)
    except Exception:
        pass

    # Approach 4: Environment variable (proxy setups)
    try:
        import os
        ip = os.getenv("REMOTE_ADDR", "")
        if ip and ip != "unknown":
            return ip
    except Exception:
        pass

    return "unknown"


# ── session init ──────────────────────────────────────────────

def _init_session():
    if "session_id" not in st.session_state:
        sid = str(uuid.uuid4())
        ip  = _get_client_ip()
        st.session_state.session_id = sid
        st.session_state.client_ip  = ip
        st.session_state.qa_turn    = 0
        activity_log.log_session_start(sid, ip)
        log.info(f"New session id={sid} ip={ip}")

    defaults = {
        "extracted_text": None,
        "image_map":      {},
        "table_count":    0,
        "chat_history":   [],
        "messages":       [],
        "doc_name":       None,
        "doc_pages":      None,
        "doc_chars":      None,
        "doc_images":     0,
        "doc_tables":     0,
        "doc_scanned":    False,
        "md_saved_path":  None,
        "qa_turn":        0,
        "uploader_key":   0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ── helpers ───────────────────────────────────────────────────

_IMAGE_REF_RE = re.compile(r"(?:see\s+)?[Ii]mage\s+(\d+)", re.IGNORECASE)


def _render_referenced_images(answer_text: str, image_map: dict):
    """
    Parse LLM answer for 'Image N' references and render those images.
    Returns list of rendered image paths.
    """
    rendered = []
    seen     = set()
    for m in _IMAGE_REF_RE.finditer(answer_text):
        n = int(m.group(1))
        if n not in seen and n in image_map:
            seen.add(n)
            img_path = image_map[n]
            if Path(img_path).exists():
                rendered.append(img_path)
                st.image(img_path, caption=f"Image {n}", width=700)
    return rendered


def _do_extraction(uploaded_file):
    """Run extraction pipeline and update session state."""

    # ── Loading bar — shown in sidebar below uploader ─────────
    loading_bar = st.sidebar.empty()
    loading_bar.markdown(
        """
        <style>
        .docqa-loading-wrap {
            width: 100%;
            height: 3px;
            background: rgba(124, 58, 237, 0.15);
            border-radius: 2px;
            overflow: hidden;
            margin: 6px 0 12px 0;
        }
        .docqa-loading-line {
            height: 100%;
            width: 40%;
            background: linear-gradient(
                90deg,
                transparent 0%,
                #7c3aed 40%,
                #a78bfa 60%,
                transparent 100%
            );
            border-radius: 2px;
            animation: docqa-slide 1.4s ease-in-out infinite;
        }
        @keyframes docqa-slide {
            0%   { transform: translateX(-100%); }
            100% { transform: translateX(350%);  }
        }
        </style>
        <div class="docqa-loading-wrap">
            <div class="docqa-loading-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    status_text = st.sidebar.empty()
    status_text.caption(f"⏳ Working on **{uploaded_file.name}** — please wait...")


    try:
        suffix = Path(uploaded_file.name).suffix
        tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.read())
        tmp.flush()
        tmp.close()

        res = extract_document(
            file_path         = tmp.name,
            file_ext          = suffix.lstrip("."),
            session_id        = st.session_state.session_id,
            original_filename = uploaded_file.name,
        )

        st.session_state.extracted_text = res.full_text
        st.session_state.image_map      = res.image_map
        st.session_state.table_count    = res.table_count
        st.session_state.doc_name       = uploaded_file.name
        st.session_state.doc_pages      = res.page_count
        st.session_state.doc_chars      = len(res.full_text)
        st.session_state.doc_images     = len(res.image_map)
        st.session_state.doc_tables     = res.table_count
        st.session_state.doc_scanned    = res.is_scanned
        st.session_state.md_saved_path  = res.md_saved_path
        st.session_state.chat_history   = []
        st.session_state.messages       = []
        st.session_state.qa_turn        = 0

        activity_log.log_document(
            filename = uploaded_file.name,
            pages    = res.page_count,
            chars    = len(res.full_text),
            images   = len(res.image_map),
            tables   = res.table_count,
        )
        activity_log.log_extraction_timing(
            t_page_check = res.t_page_check,
            t_text       = res.t_text_extraction,
            t_images     = res.t_image_extraction,
            t_total      = res.t_total,
            t_tables     = res.t_table_extraction,
        )

        log.info(f"Extracted: {uploaded_file.name} | {res.summary}")
        loading_bar.empty()
        status_text.empty()
        st.success("✅ Document ready — ask your question below!")

    except ValueError as exc:
        loading_bar.empty()
        status_text.empty()
        st.error(str(exc))
        log.warning(str(exc))
    except Exception as exc:
        loading_bar.empty()
        status_text.empty()
        st.error(f"Extraction error: {exc}")
        log.exception(str(exc))


def _clear_document():
    """Reset all document state and refresh uploader widget."""
    for k, v in {
        "extracted_text": None,
        "image_map":      {},
        "table_count":    0,
        "doc_name":       None,
        "doc_pages":      None,
        "doc_chars":      None,
        "doc_images":     0,
        "doc_tables":     0,
        "doc_scanned":    False,
        "md_saved_path":  None,
        "chat_history":   [],
        "messages":       [],
        "qa_turn":        0,
    }.items():
        st.session_state[k] = v
    st.session_state["uploader_key"] += 1
    log.info("Document cleared by user")


# ── sidebar — upload + doc info ──────────────────────────────

with st.sidebar:

    # ── Upload section ─────────────────────────────────────────
    st.subheader("📂 Upload Document")
    st.caption("PDF · DOCX · TXT — max 50 pages")

    uploaded_file = st.file_uploader(
        label            = "Choose a file",
        type             = SUPPORTED_EXTENSIONS,
        label_visibility = "collapsed",
        key              = f"uploader_{st.session_state.uploader_key}",
    )

    # Trigger extraction when new file uploaded
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.doc_name:
            _do_extraction(uploaded_file)

    # ── Clear button ───────────────────────────────────────────
    st.button(
        "🗑️ Clear Document",
        use_container_width = True,
        disabled            = not st.session_state.doc_name,
        help                = "Clear the current document and reset chat",
        on_click            = _clear_document,
    )

    st.divider()

    # ── Document info — shown after extraction ─────────────────
    if st.session_state.doc_name:
        st.subheader("📋 Document Info")
        st.markdown(f"**File:** {st.session_state.doc_name}")
        st.markdown(f"**Pages:** {st.session_state.doc_pages}")
        st.markdown(f"**Characters:** {st.session_state.doc_chars:,}")

        img_count = st.session_state.doc_images
        tbl_count = st.session_state.doc_tables
        st.markdown(
            f"**Images found:** {img_count}" + (" ✅" if img_count else "")
        )
        st.markdown(
            f"**Tables found:** {tbl_count}" + (" ✅" if tbl_count else "")
        )

        if st.session_state.doc_scanned:
            st.caption("🔍 Scanned PDF — EasyOCR was used")
        else:
            st.caption("📝 Digital document")

        if st.session_state.md_saved_path:
            st.caption(f"💾 Saved: `{st.session_state.md_saved_path}`")


# ══════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════

# ── 1. Heading ────────────────────────────────────────────────

st.title("🤖 AI-Powered Document Assistant")

# ── 2. About + Disclaimer box ─────────────────────────────────

with st.container(border=True):
    col_about, col_disclaimer = st.columns(2)

    with col_about:
        st.markdown("**About**")
        st.markdown(
            "This tool allows you to ask questions from any uploaded document — "
            "PDF, DOCX, or TXT — and receive precise answers powered by a local "
            "AI vision model. It understands text, tables, and images."
        )

    with col_disclaimer:
        st.markdown("**Disclaimer**")
        st.markdown(
            "Answers are generated strictly from the uploaded document. "
            "The system does not use internet or external knowledge. "
            "Always verify critical information from the original source."
        )

st.divider()

# ── 4. Chat messages ──────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for img_path in msg.get("images", []):
            if Path(img_path).exists():
                st.image(img_path, width=700)
st.markdown(
    """
    <style>
    .footer {
        position  : fixed;
        bottom    : 0;
        left      : 0;
        width     : 100%;
        text-align: center;
        font-size : 12px;
        color     : rgba(150,150,150,0.6);
        padding   : 6px 0;
        background: #ffffff;
        z-index   : 9999;
    }
    </style>
    <div class="footer">
        Designed &amp; Developed by Dept. of CSE &nbsp;|&nbsp; Support Contact: 3000
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .footer {
        position  : fixed;
        bottom    : 0;
        left      : 0;
        width     : 100%;
        text-align: center;
        font-size : 12px;
        color     : rgba(150,150,150,0.6);
        padding   : 6px 0;
        background: #ffffff;
        z-index   : 9999;
    }
    </style>
    <div class="footer">
        Designed &amp; Developed by Dept. of CSE &nbsp;|&nbsp; Support Contact: 3000
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 5. Chat input ─────────────────────────────────────────────

question = st.chat_input(
    placeholder = "Ask about text, tables, charts, or images in your document…",
    disabled    = not st.session_state.extracted_text,
)

if question:
    log.info(f"Q: {question[:120]}")

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({
        "role": "user", "content": question, "images": [], "mode": None,
    })

    # ── Build prompt ───────────────────────────────────────────
    prompt, images_b64, t_prompt, qtype = build_prompt(
        extracted_text = st.session_state.extracted_text,
        chat_history   = st.session_state.chat_history,
        question       = question,
        image_map      = st.session_state.image_map,
    )
    stats = GenerationStats(t_prompt_build=t_prompt)

    log.info(
        f"Prompt: {len(prompt):,} chars | mode={qtype.mode} | "
        f"{len(images_b64)} image(s) | build={t_prompt:.3f}s"
    )

    # ── Stream answer ──────────────────────────────────────────
    full_answer       = ""
    referenced_images = []

    with st.chat_message("assistant"):
        try:
            full_answer = st.write_stream(
                stream_response(prompt, images_b64, stats)
            )
        except Exception as exc:
            full_answer = (
                f"❌ Ollama error: `{exc}`\n\n"
                "Check that Ollama is running and OLLAMA_HOST in config.py is correct."
            )
            st.error(full_answer)
            log.error(f"Ollama error: {exc}")

        # Render images referenced in the answer ("See Image N")
        if st.session_state.image_map:
            referenced_images = _render_referenced_images(
                full_answer, st.session_state.image_map
            )

    # ── Update session state ───────────────────────────────────
    st.session_state.qa_turn += 1
    st.session_state.chat_history.append({
        "question": question,
        "answer":   full_answer,
    })
    if len(st.session_state.chat_history) > MAX_HISTORY_TURNS:
        st.session_state.chat_history = (
            st.session_state.chat_history[-MAX_HISTORY_TURNS:]
        )

    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_answer,
        "images":  referenced_images,
        "mode":    qtype.mode,
    })

    # ── Activity log ───────────────────────────────────────────
    activity_log.log_qa(
        turn          = st.session_state.qa_turn,
        question      = question,
        answer        = full_answer,
        t_prompt      = stats.t_prompt_build,
        t_ttft        = stats.t_ttft,
        t_total       = stats.t_total,
        chars_per_sec = stats.chars_per_sec,
        question_type = qtype.mode,
        images_sent   = len(images_b64),
    )

    log.info(
        f"QA done | Turn {st.session_state.qa_turn} | "
        f"mode={qtype.mode} | {stats.summary}"
    )



