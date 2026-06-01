
import re
import uuid
import tempfile
from pathlib import Path

import streamlit as st
import random

from config import SUPPORTED_EXTENSIONS, MAX_HISTORY_TURNS, MAX_IMAGES_PER_QUERY, DOCUMENT_SUMMARY
from extractor import extract_document, cleanup_session_images
from ollama_connect import (
    build_prompt,
    build_summary_prompt,
    check_ollama_connection,
    stream_response,
    GenerationStats,
)
from logger import get_logger, activity_log

log = get_logger("app")


# ── page config ───────────────────────────────────────────────

st.set_page_config(
    page_title = "Document Assistant",
    page_icon  = "📄",
    layout     = "wide",
)


# ── Ollama check — terminal/log only,─────────

@st.cache_data(ttl=60, show_spinner=False)
def _cached_ollama_check() -> tuple:
    return check_ollama_connection()

_ok, _info = _cached_ollama_check()
log.info(f"Ollama | {'Connected' if _ok else 'NOT reachable'} | {_info}")


# ── client IP detection ───────────────────────────────────────
def _get_client_ip() -> str:
    try:
        from streamlit import runtime
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        if ctx is None:
            return "unknown"
        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return "unknown"
        return f"{session_info.request.remote_ip}"
    except Exception:
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
        "_streaming":     False,
        "summary_done":   False,   # ← guard: summary runs exactly once per document
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
    status_text.caption(f" I'm Working on **{uploaded_file.name}** — Please wait...")


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
        st.success(" I'M Ready to Answer Your Questions ")

        # Summary will be triggered at module level on the next rerun,
        # guarded by summary_done flag — do NOT call it here (sidebar context).

    except ValueError as exc:
        loading_bar.empty()
        status_text.empty()
        st.session_state["uploader_key"] += 1
        st.error(str(exc))
        log.warning(str(exc))
    except Exception as exc:
        loading_bar.empty()
        status_text.empty()
        st.session_state["uploader_key"] += 1
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
        "summary_done":   False,   # ← reset so next document gets a fresh summary
    }.items():
        st.session_state[k] = v
    st.session_state["uploader_key"] += 1
    log.info("Document cleared by user")


# ── sidebar — upload + doc info ──────────────────────────────

with st.sidebar:

    # ── Upload section ─────────────────────────────────────────
    st.subheader("📂 Upload Document")
    st.caption("PDF · DOCX · TXT — max 100 pages")

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
        " Clear Document",
        use_container_width = True,
        disabled            = not st.session_state.doc_name,
        help                = "Clear the current document and reset chat",
        on_click            = _clear_document,
    )

    st.divider()

    # ── Document info — shown after extraction ─────────────────
    if st.session_state.doc_name and not st.session_state._streaming:
        st.subheader(" Document Info")
        st.markdown(f"**File:** {st.session_state.doc_name}")
        st.markdown(f"**Pages:** {st.session_state.doc_pages}")
        #st.markdown(f"**Characters:** {st.session_state.doc_chars:,}")

        img_count = st.session_state.doc_images
        tbl_count = st.session_state.doc_tables
        st.markdown(
            f"**Images found:** {img_count}" + (" " if img_count else "")
        )
        st.markdown(
            f"**Tables found:** {tbl_count}" + (" " if tbl_count else "")
        )

        if st.session_state.doc_scanned:
            st.caption(" Scanned PDF ")
        else:
            st.caption(" Digital document")

        #if st.session_state.md_saved_path:
            #st.caption(f" Saved: `{st.session_state.md_saved_path}`")


# ══════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════

# ── 1. Heading ────────────────────────────────────────────────

st.title("AI Based Document Assistant")

# ── 2. About + Disclaimer box ─────────────────────────────────

with st.container(border=True):
    col_about, col_disclaimer = st.columns(2)

    with col_about:
        st.markdown("**About**")
        st.markdown(
            "This application allows you to ask questions from any uploaded document — "
            "PDF, DOCX, or TXT from SideBar. "
            " It understands text, tables, and images."
        )

    with col_disclaimer:
        st.markdown("**Disclaimer**")
        st.markdown(
            "Answers are generated strictly from the uploaded document. "
            "The system does not use external knowledge. "
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



def _generate_summary():
    """
    Generate a detailed document summary using the LLM and display it
    as the first assistant message in the chat.
    Stores the summary in chat_history so Q&A turns are aware of it.
    Only called when DOCUMENT_SUMMARY = True in config.
    Runs exactly once per document, guarded by st.session_state.summary_done.
    """
    prompt, t_build = build_summary_prompt(st.session_state.extracted_text)
    stats           = GenerationStats(t_prompt_build=t_build)

    log.info(
        f"[Summary] Starting | {len(prompt):,} chars | build={t_build:.3f}s"
    )

    summary_text = ""
    st.session_state["_streaming"] = True

    with st.spinner("Generating document summary…"):
        try:
            with st.chat_message("assistant"):
                summary_text = st.write_stream(
                    stream_response(prompt, [], stats)
                )
        except Exception as exc:
            summary_text = f"⚠️ Summary generation failed: `{exc}`"
            with st.chat_message("assistant"):
                st.error(summary_text)
            log.error(f"[Summary] Error: {exc}")

    st.session_state["_streaming"] = False
    st.session_state["summary_done"] = True   # ← mark done so it never runs again

    # Store in chat_history (LLM context) but not in messages (UI display)
    st.session_state.chat_history.append({
        "question": "Please summarize this document.",
        "answer":   summary_text,
    })

    log.info(
        f"[Summary] Done | {stats.summary}"
    )

# Trigger summary exactly once: doc loaded, not yet summarised, not mid-stream
if (
    DOCUMENT_SUMMARY
    and st.session_state.doc_name
    and not st.session_state.summary_done
    and not st.session_state._streaming
):
    _generate_summary()

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

    THINKING_MESSAGES = [
        "Analyzing your query against the available documents.",
        "Retrieving the most relevant information.",
        "Processing context to ensure an accurate response.",
        "Searching the knowledge base for matching insights.",
        "Reviewing related content to form a precise answer.",
        "Evaluating document sections for relevance.",
        "Synthesizing information from multiple sources.",
        "Aligning your question with stored knowledge.",
        "Identifying key details required for the response.",
        "Cross-checking facts to improve accuracy.",
        "Understanding the intent behind your question.",
        "Filtering unnecessary information.",
        "Assembling a clear and concise answer.",
        "Validating results before responding.",
        "Matching your query with indexed content.",
        "Optimizing the response for clarity.",
        "Finalizing the answer for delivery.",
    ]

    st.session_state["_streaming"] = True
    with st.spinner(random.choice(THINKING_MESSAGES)):
        try:
            full_answer = st.write_stream(
                stream_response(prompt, images_b64, stats)
            )
        except Exception as exc:
            full_answer = (
                f" Ollama error: `{exc}`\n\n"
                "Check Ollama is running and OLLAMA_HOST in config.py is correct."
            )
            st.error(full_answer)
            log.error(f"Ollama error: {exc}")

            # Show assistant message
            with st.chat_message("assistant"):
                st.markdown(full_answer)

        # Render images referenced in the answer ("See Image N")
        if st.session_state.image_map:
            referenced_images = _render_referenced_images(
                full_answer, st.session_state.image_map
            )
        st.session_state["_streaming"] = False
        

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
        ip            = st.session_state.client_ip,
        
        
    )

    log.info(
        f"QA done | Turn {st.session_state.qa_turn} | "
        f"mode={qtype.mode} | {stats.summary}"
    )

footer_html = """
<style>
/* Container that holds the footer text */
footer {
    position: fixed;          /* Keep it fixed relative to the viewport */
    left: 0;
    bottom: 0;
    width: 100%;              /* Span the full width of the page */
    background-color: #f8fafc;/* Light background (optional) */
    color: #64748b;           /* Text colour – matches your original style */
    text-align: center;
    font-size: 10px;
    padding: 0 0;           /* Vertical spacing */
    z-index: 999;             /* Ensure it appears above other elements */
}
</style>

<footer>

Designed & Developed by DAIV, RCI Team   -For Support Call 040-2430 7192

</footer>
"""

st.markdown(footer_html, unsafe_allow_html=True)



