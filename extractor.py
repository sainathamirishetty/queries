"""
extractor.py
─────────────────────────────────────────────────────────────
Document extraction pipeline for DocQA.

Supports:
  • PDF  — Marker for digital + scanned pages (replaces pdfplumber + EasyOCR)
           Marker handles: multi-column, tables, figures, equations, OCR
           internally. No separate OCR fallback needed.
  • DOCX — python-docx for paragraphs + tables (→ markdown)
  • TXT  — plain UTF-8 read

Key design:
  • Marker produces clean structured markdown from PDFs directly.
  • Images are extracted from the Marker output, saved to disk, and
    referenced as [Image N appears here] markers inside the text.
  • Tables come out as proper pipe-markdown from Marker — no custom
    table parser needed for PDFs.
  • EasyOCR is kept ONLY as a last-resort fallback if Marker is not
    installed. When EasyOCR IS used (fallback path), the reader is
    instantiated once as a module-level singleton — not per page.
─────────────────────────────────────────────────────────────
"""

import io
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import (
    EXTRACTED_DOCS_DIR,
    MAX_PAGES,
    SCANNED_TEXT_THRESHOLD,
)
from logger import get_logger

log = get_logger("extractor")


# ── EasyOCR singleton (fallback only — not used when Marker is installed) ─────
# Instantiated once on first use, reused for every subsequent scanned page.
# This fixes the original bug where easyocr.Reader() was called inside the
# per-page loop, paying the 2-5s model-load cost on every single page.

_easyocr_reader = None

def _get_easyocr_reader():
    """Return the shared EasyOCR reader, initialising it only once."""
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            log.info("Initialising EasyOCR reader (one-time cost)…")
            _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            log.info("EasyOCR reader ready.")
        except ImportError:
            log.warning("easyocr not installed — scanned pages will be blank.")
            _easyocr_reader = None
    return _easyocr_reader


# ── result dataclass ──────────────────────────────────────────

@dataclass
class ExtractionResult:
    full_text:           str              = ""
    image_map:           Dict[int, str]   = field(default_factory=dict)
    table_count:         int              = 0
    page_count:          int              = 0
    is_scanned:          bool             = False
    md_saved_path:       Optional[str]    = None
    t_page_check:        float            = 0.0
    t_text_extraction:   float            = 0.0
    t_image_extraction:  float            = 0.0
    t_table_extraction:  float            = 0.0
    t_total:             float            = 0.0

    @property
    def summary(self) -> str:
        return (
            f"Pages: {self.page_count} | Chars: {len(self.full_text):,} | "
            f"Images: {len(self.image_map)} | Tables: {self.table_count} | "
            f"Scanned: {self.is_scanned} | Total: {self.t_total:.2f}s"
        )


# ── shared helpers ────────────────────────────────────────────

def _make_output_dir(original_filename: str) -> Path:
    date_str  = datetime.now().strftime("%Y-%m-%d")
    stem      = Path(original_filename).stem
    safe_stem = re.sub(r"[^\w\-]", "_", stem)[:60]
    out_dir   = Path(EXTRACTED_DOCS_DIR) / date_str / safe_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _table_to_markdown(table: List[List]) -> str:
    """
    Convert a list-of-rows table to GitHub-Flavoured Markdown pipe format.
    Used by DOCX extractor. Marker produces its own markdown tables for PDFs.
    """
    if not table or not table[0]:
        return ""

    rows = []
    for row in table:
        normalised = [str(c).strip() if c is not None else "" for c in row]
        if any(normalised):
            rows.append(normalised)

    if not rows:
        return ""

    col_count = max(len(r) for r in rows)
    rows = [r + [""] * (col_count - len(r)) for r in rows]

    col_widths = [
        max(max(len(rows[i][c]) for i in range(len(rows))), 3)
        for c in range(col_count)
    ]

    def _fmt(row):
        return "| " + " | ".join(row[c].ljust(col_widths[c]) for c in range(col_count)) + " |"

    header    = _fmt(rows[0])
    separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
    body      = [_fmt(r) for r in rows[1:]]

    return "\n".join([header, separator] + body)


def _save_image_bytes(img_bytes: bytes, out_dir: Path, img_num: int, ext: str = "png") -> str:
    img_path = out_dir / f"image_{img_num:03d}.{ext}"
    img_path.write_bytes(img_bytes)
    return str(img_path)


def _pil_to_png_bytes(pil_image) -> bytes:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def _save_markdown(text: str, out_dir: Path, stem: str) -> str:
    md_path = out_dir / f"{stem}.md"
    md_path.write_text(text, encoding="utf-8")
    return str(md_path)


def _count_pipe_tables(text: str) -> int:
    """Count markdown pipe tables in extracted text."""
    # A table block has at least one separator row: |---|---|
    return len(re.findall(r"^\|[-| :]+\|$", text, re.MULTILINE))


# ── PDF extraction via MinerU ─────────────────────────────────

def _extract_pdf_mineru(file_path: str, out_dir: Path) -> Tuple[
    str, Dict[int, str], int, bool, float, float, float, float
]:
    """
    Extract PDF using MinerU magic-pdf 0.9.x — fully offline.

    Confirmed API from terminal output:
        parse_union_pdf(pdf_bytes, pdf_models, imageWriter)
        parse_ocr_pdf(pdf_bytes, pdf_models, imageWriter)

    pdf_models must be loaded first via doc_analyze().
    doc_analyze() reads model config from ~/.mineru/magic-pdf.json
    and loads all models from the local models directory.
    No internet calls at runtime.

    Flow:
        1. Read PDF bytes
        2. doc_analyze() → loads models + classifies pages → returns model list
        3. parse_union_pdf() → processes PDF → returns page dict list
        4. dict2md → converts page dicts to markdown string
        5. Harvest images, inject markers, add page markers
    """
    import shutil
    import tempfile

    from magic_pdf.user_api import parse_union_pdf, parse_ocr_pdf, doc_analyze
    from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter

    # ── Step 1: read PDF bytes ────────────────────────────────
    t0 = time.perf_counter()
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()
    t_page_check = time.perf_counter() - t0

    with tempfile.TemporaryDirectory() as tmp_work:
        img_work_dir = os.path.join(tmp_work, "images")
        os.makedirs(img_work_dir, exist_ok=True)

        # ── Build LocalWriter (AbsReaderWriter subclass) ──────
        # Confirmed from terminal: AbsReaderWriter is in user_api namespace
        # meaning it IS the base class we need to subclass.
        # We implement all methods that could be called during processing.
        class LocalWriter(AbsReaderWriter):
            def __init__(self, base_dir):
                self.base_dir = base_dir
                os.makedirs(base_dir, exist_ok=True)

            def write(self, content, path, mode="text"):
                full = os.path.join(self.base_dir, path)
                os.makedirs(os.path.dirname(os.path.abspath(full)), exist_ok=True)
                if isinstance(content, bytes) or mode == "binary":
                    with open(full, "wb") as f:
                        f.write(content if isinstance(content, bytes) else content.encode())
                else:
                    with open(full, "w", encoding="utf-8") as f:
                        f.write(content)

            def read(self, path, mode="text"):
                full = os.path.join(self.base_dir, path)
                if not os.path.exists(full):
                    return b"" if mode == "binary" else ""
                if mode == "binary":
                    with open(full, "rb") as f:
                        return f.read()
                with open(full, "r", encoding="utf-8") as f:
                    return f.read()

            def read_offset(self, path, offset=None, limit=None):
                return self.read(path, mode="binary")

        image_writer = LocalWriter(img_work_dir)

        # ── Step 2: doc_analyze — load models + classify ──────
        # doc_analyze() is the key function we were missing.
        # It reads ~/.mineru/magic-pdf.json, loads all local models,
        # classifies each page (digital/scanned), and returns the
        # model list that parse_union_pdf requires.
        t0 = time.perf_counter()
        log.info("MinerU: running doc_analyze (loading models)...")
        try:
            pdf_models = doc_analyze(
                pdf_bytes,
                ocr=False,          # False = auto-detect, True = force OCR
            )
            is_scanned = False
            log.info(f"MinerU doc_analyze OK | models loaded: {len(pdf_models)} pages")
        except Exception as e:
            log.warning(f"doc_analyze(ocr=False) failed ({e}), trying ocr=True...")
            try:
                pdf_models = doc_analyze(pdf_bytes, ocr=True)
                is_scanned = True
                log.info(f"MinerU doc_analyze OCR OK")
            except Exception as e2:
                raise RuntimeError(
                    f"doc_analyze failed.\n"
                    f"non-ocr error: {e}\nocr error: {e2}"
                )

        # ── Step 3: parse PDF → page dict list ───────────────
        log.info("MinerU: running parse_union_pdf...")
        try:
            pdf_info_list = parse_union_pdf(
                pdf_bytes,
                pdf_models,
                image_writer,
            )
            log.info(f"MinerU parse_union_pdf OK | pages={len(pdf_info_list)}")
        except Exception as e1:
            log.warning(f"parse_union_pdf failed ({e1}), trying parse_ocr_pdf...")
            try:
                pdf_info_list = parse_ocr_pdf(
                    pdf_bytes,
                    pdf_models,
                    image_writer,
                )
                is_scanned = True
                log.info(f"MinerU parse_ocr_pdf OK | pages={len(pdf_info_list)}")
            except Exception as e2:
                raise RuntimeError(
                    f"Both parse modes failed.\n"
                    f"union: {e1}\nocr: {e2}"
                )

        t_text = time.perf_counter() - t0

        t_text = time.perf_counter() - t0

        # ── Step 4: convert page dicts to markdown ────────────
        t0 = time.perf_counter()
        try:
            from magic_pdf.dict2md.ocr_mkcontent import union_make
            raw_markdown = union_make(pdf_info_list)
        except TypeError:
            # some versions need different args — try alternatives
            try:
                from magic_pdf.dict2md.ocr_mkcontent import union_make
                raw_markdown = union_make(
                    pdf_info_list,
                    make_mode="mm",
                    drop_reason_flag=False,
                )
            except Exception:
                raw_markdown = _mineru_manual_text(pdf_info_list)
        except Exception as exc:
            log.warning(f"dict2md failed ({exc}), using manual text extraction")
            raw_markdown = _mineru_manual_text(pdf_info_list)

        log.info(f"MinerU markdown: {len(raw_markdown):,} chars")

        # ── Step 5: harvest images ────────────────────────────
        image_map:   Dict[int, str] = {}
        img_counter                 = 0
        processed_text              = raw_markdown

        # MinerU places images in markdown as: ![](images/filename.png)
        img_ref_pattern = re.compile(r"!\[([^\]]*)\]\(images/([^)]+)\)")

        for m in img_ref_pattern.finditer(raw_markdown):
            original_ref = m.group(0)
            img_filename = m.group(2)
            src_path     = os.path.join(img_work_dir, img_filename)

            if os.path.exists(src_path):
                img_counter += 1
                dst_path = str(out_dir / f"image_{img_counter:03d}.png")
                shutil.copy2(src_path, dst_path)
                image_map[img_counter] = dst_path
                replacement  = f"\n[Image {img_counter} appears here]\n"
                processed_text = processed_text.replace(original_ref, replacement, 1)
                log.debug(f"Saved MinerU image {img_counter}: {img_filename}")
            else:
                log.warning(f"MinerU image not found: {src_path}")

        t_images = time.perf_counter() - t0

        # ── Step 6: add page markers ──────────────────────────
        t0 = time.perf_counter()
        page_num    = 1
        paged_lines = [f"\n<!-- Page {page_num} -->\n"]

        for line in processed_text.splitlines():
            stripped = line.strip()
            is_page_break = (
                stripped == "---"
                or stripped == "___"
                or "MD_PRINT_STYLE_END" in stripped
                or "page_break" in stripped.lower()
            )
            if is_page_break:
                page_num += 1
                paged_lines.append(f"\n<!-- Page {page_num} -->\n")
            else:
                paged_lines.append(line)

        final_text  = "\n".join(paged_lines)
        table_count = _count_pipe_tables(final_text)
        t_tables    = time.perf_counter() - t0

    log.info(
        f"MinerU done | Pages: {page_num} | Tables: {table_count} | "
        f"Images: {img_counter} | Scanned: {is_scanned}"
    )

    return (
        final_text, image_map, table_count, is_scanned,
        t_page_check, t_text, t_images, t_tables,
    )


def _mineru_manual_text(pdf_info_list: list) -> str:
    """
    Fallback: extract plain text from MinerU page dict list
    when the dict2md converter is unavailable or fails.
    Walks the nested dict structure and collects all text spans.
    """
    lines = []
    for page_idx, page_info in enumerate(pdf_info_list):
        lines.append(f"\n<!-- Page {page_idx + 1} -->\n")
        # MinerU page dict has 'preproc_blocks' or 'para_blocks'
        blocks = (
            page_info.get("para_blocks")
            or page_info.get("preproc_blocks")
            or []
        )
        for block in blocks:
            block_type = block.get("type", "")
            # text block
            if "text" in block_type.lower() or block_type == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("content", "").strip()
                        if text:
                            lines.append(text)
            # table block
            elif "table" in block_type.lower() or block_type == 3:
                table_body = block.get("html", "") or block.get("latex", "")
                if table_body:
                    lines.append(f"\n{table_body}\n")
            # title/heading block
            elif "title" in block_type.lower() or block_type == 1:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("content", "").strip()
                        if text:
                            lines.append(f"\n## {text}\n")
    return "\n".join(lines)


# ── PDF fallback — pdfplumber + EasyOCR singleton ────────────
# Used ONLY when Marker is not installed.
# EasyOCR reader is now a module-level singleton (see top of file).

def _ocr_page_fallback(pil_image) -> str:
    """
    OCR a PIL image using the singleton EasyOCR reader.
    Reader is initialised once and reused — NOT created per page.
    """
    try:
        import numpy as np
        reader = _get_easyocr_reader()
        if reader is None:
            return "[scanned page — install easyocr for OCR]"
        result = reader.readtext(np.array(pil_image), detail=0, paragraph=True)
        return "\n".join(result)
    except Exception as exc:
        log.warning(f"OCR error: {exc}")
        return "[OCR failed]"


def _is_scanned_page(page, raw_text: str) -> bool:
    """
    Multi-signal scanned page detection. Replaces the fragile single
    character-count threshold that caused false positives on sparse
    but fully digital pages (title-only pages, figure-only pages, etc.)

    A page is scanned ONLY when ALL of these are true:
      1. Very few extractable characters (below threshold)
      2. Zero character/font objects on the page (page.chars is empty)
      3. At least one embedded image exists on the page

    Digital pages always have font objects even when sparse.
    Scanned pages have image pixels but zero font objects.
    """
    # Signal 1: sufficient text already extracted → definitely digital
    if len(raw_text.strip()) >= SCANNED_TEXT_THRESHOLD:
        return False

    # Signal 2: font/character objects present → digital, just sparse
    # page.chars returns all character-level objects pdfplumber found
    if len(page.chars) > 0:
        return False

    # Signal 3: images present + zero chars → true scanned page
    if len(page.images) > 0 and len(page.chars) == 0:
        return True

    # Default: don't OCR — safer to return empty text than garbage OCR
    return False


def _extract_pdf_fallback(file_path: str, out_dir: Path) -> Tuple[
    str, Dict[int, str], int, bool, float, float, float, float
]:
    """
    Fallback PDF extractor using pdfplumber + EasyOCR singleton.
    Called only when Marker is not installed.

    Key improvement over original:
      - _is_scanned_page() uses multi-signal detection (not just char count)
      - EasyOCR reader is a singleton — not re-instantiated per page
    """
    import pdfplumber

    pages_text: List[str]      = []
    image_map:  Dict[int, str] = {}
    img_counter                = 0
    table_count                = 0
    is_scanned                 = False
    t_page_check = t_text = t_images = t_tables = 0.0

    with pdfplumber.open(file_path) as pdf:
        total_pages = min(len(pdf.pages), MAX_PAGES)

        for page_num, page in enumerate(pdf.pages[:total_pages], start=1):
            log.debug(f"PDF page {page_num}/{total_pages}")
            parts: List[str] = [f"\n<!-- Page {page_num} -->\n"]

            # ── get raw text ──
            t0 = time.perf_counter()
            raw_text = page.extract_text() or ""
            t_page_check += time.perf_counter() - t0

            # ── multi-signal scanned detection + OCR ──
            t0 = time.perf_counter()
            if _is_scanned_page(page, raw_text):
                is_scanned = True
                log.info(f"Page {page_num}: confirmed scanned — running OCR")
                # 300 DPI for better OCR accuracy (was 150 before)
                pil_img  = page.to_image(resolution=300).original
                raw_text = _ocr_page_fallback(pil_img)
            else:
                log.debug(f"Page {page_num}: digital (chars={len(page.chars)})")
            parts.append(raw_text)
            t_text += time.perf_counter() - t0

            # ── table extraction → markdown ──
            t0 = time.perf_counter()
            try:
                page_tables = page.extract_tables() or []
                for tbl in page_tables:
                    md = _table_to_markdown(tbl)
                    if md:
                        table_count += 1
                        parts.append(
                            f"\n\n<!-- Table {table_count} (Page {page_num}) -->\n"
                            + md + "\n"
                        )
            except Exception as exc:
                log.warning(f"Table extraction failed page {page_num}: {exc}")
            t_tables += time.perf_counter() - t0

            # ── image extraction ──
            t0 = time.perf_counter()
            try:
                for img_info in page.images:
                    try:
                        x0, y0 = img_info["x0"], img_info["y0"]
                        x1, y1 = img_info["x1"], img_info["y1"]
                        cropped = page.within_bbox((x0, y0, x1, y1)).to_image(
                            resolution=150
                        ).original
                        img_counter += 1
                        img_path = _save_image_bytes(
                            _pil_to_png_bytes(cropped), out_dir, img_counter
                        )
                        image_map[img_counter] = img_path
                        parts.append(f"\n[Image {img_counter} appears here]\n")
                        log.debug(f"Saved image {img_counter} from page {page_num}")
                    except Exception as exc:
                        log.warning(f"Image on page {page_num} failed: {exc}")
            except Exception as exc:
                log.warning(f"Image loop failed page {page_num}: {exc}")
            t_images += time.perf_counter() - t0

            pages_text.append("\n".join(parts))

    return (
        "\n".join(pages_text), image_map, table_count, is_scanned,
        t_page_check, t_text, t_images, t_tables,
    )


def _extract_pdf(file_path: str, out_dir: Path) -> Tuple[
    str, Dict[int, str], int, bool, float, float, float, float
]:
    """
    PDF extraction dispatcher.
    Tries MinerU first (best quality, fully offline).
    Falls back to pdfplumber + EasyOCR singleton if MinerU is not installed.
    """
    try:
        import magic_pdf  # noqa: F401
        log.info("Using MinerU for PDF extraction (offline, high quality).")
        return _extract_pdf_mineru(file_path, out_dir)
    except ImportError:
        log.warning(
            "MinerU (magic-pdf) not installed — falling back to pdfplumber + EasyOCR. "
            "pip install magic-pdf==0.9.3"
        )
        return _extract_pdf_fallback(file_path, out_dir)


# ── DOCX extraction ───────────────────────────────────────────

def _extract_docx(file_path: str, out_dir: Path) -> Tuple[
    str, Dict[int, str], int, float, float, float
]:
    """
    Returns: full_text, image_map, table_count, t_text, t_images, t_tables
    Iterates body XML children in order so paragraphs and tables appear in
    their original reading sequence.
    """
    from docx import Document

    doc = Document(file_path)

    parts:       List[str]      = ["<!-- Page 1 -->\n"]
    image_map:   Dict[int, str] = {}
    img_counter                 = 0
    table_count                 = 0
    saved_rids:  Dict[str, int] = {}

    t_text = t_images = t_tables = 0.0

    tbl_lookup = {t._tbl: t for t in doc.tables}

    NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

    def _get_inline_image_rids(para_xml) -> List[str]:
        rids = []
        for elem in para_xml.iter():
            local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if local == "blip":
                rid = elem.get(f"{{{NS_REL}}}embed")
                if rid:
                    rids.append(rid)
        return rids

    for child in doc.element.body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        if tag == "p":
            t0 = time.perf_counter()
            from docx.text.paragraph import Paragraph as _Para
            para = _Para(child, doc)
            text = para.text.strip()
            if text:
                if para.style.name.startswith("Heading"):
                    lvl = "".join(c for c in para.style.name if c.isdigit()) or "1"
                    parts.append(f"\n{'#' * int(lvl)} {text}\n")
                else:
                    parts.append(text)
            t_text += time.perf_counter() - t0

            t0 = time.perf_counter()
            for rid in _get_inline_image_rids(child):
                if rid not in saved_rids:
                    try:
                        img_bytes = doc.part.related_parts[rid].blob
                        img_counter += 1
                        img_path = _save_image_bytes(img_bytes, out_dir, img_counter)
                        image_map[img_counter] = img_path
                        saved_rids[rid] = img_counter
                        parts.append(f"\n[Image {img_counter} appears here]\n")
                        log.debug(f"Saved DOCX image {img_counter}")
                    except Exception as exc:
                        log.warning(f"DOCX image {rid} failed: {exc}")
            t_images += time.perf_counter() - t0

        elif tag == "tbl":
            t0      = time.perf_counter()
            tbl_obj = tbl_lookup.get(child)
            if tbl_obj:
                rows = [
                    [cell.text.strip() for cell in row.cells]
                    for row in tbl_obj.rows
                ]
                md = _table_to_markdown(rows)
                if md:
                    table_count += 1
                    parts.append(
                        f"\n\n<!-- Table {table_count} -->\n" + md + "\n"
                    )
            t_tables += time.perf_counter() - t0

    return "\n".join(parts), image_map, table_count, t_text, t_images, t_tables


# ── TXT extraction ────────────────────────────────────────────

def _extract_txt(file_path: str) -> str:
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return "<!-- Page 1 -->\n" + Path(file_path).read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode text file with any supported encoding.")


# ── public API ────────────────────────────────────────────────

def extract_document(
    file_path:         str,
    file_ext:          str,
    session_id:        str,
    original_filename: str,
) -> ExtractionResult:
    """
    Main entry point. Dispatches to the right extractor, saves outputs,
    and returns a fully-populated ExtractionResult.
    """
    t_wall   = time.perf_counter()
    file_ext = file_ext.lower().lstrip(".")

    if file_ext not in ("pdf", "docx", "txt"):
        raise ValueError(f"Unsupported file type: .{file_ext}")

    out_dir = _make_output_dir(original_filename)
    stem    = Path(original_filename).stem
    res     = ExtractionResult()

    if file_ext == "pdf":
        (
            res.full_text,
            res.image_map,
            res.table_count,
            res.is_scanned,
            res.t_page_check,
            res.t_text_extraction,
            res.t_image_extraction,
            res.t_table_extraction,
        ) = _extract_pdf(file_path, out_dir)
        res.page_count = len(re.findall(r"<!-- Page \d+ -->", res.full_text))

    elif file_ext == "docx":
        (
            res.full_text,
            res.image_map,
            res.table_count,
            res.t_text_extraction,
            res.t_image_extraction,
            res.t_table_extraction,
        ) = _extract_docx(file_path, out_dir)
        # Estimate page count from word count (DOCX has no native page concept)
        from config import WORDS_PER_PAGE
        word_count     = len(res.full_text.split())
        res.page_count = max(1, round(word_count / WORDS_PER_PAGE))
        res.t_page_check = 0.0

    elif file_ext == "txt":
        t0                    = time.perf_counter()
        res.full_text         = _extract_txt(file_path)
        res.t_text_extraction = time.perf_counter() - t0
        res.page_count        = 1

    if res.page_count > MAX_PAGES:
        raise ValueError(
            f"Document has {res.page_count} pages — limit is {MAX_PAGES}."
        )

    res.md_saved_path = _save_markdown(res.full_text, out_dir, stem)
    res.t_total       = time.perf_counter() - t_wall

    log.info(f"Extracted '{original_filename}' → {res.summary}")
    return res


def cleanup_session_images(session_id: str):
    """Images stored permanently under extracted_docs/ — no-op by design."""
    log.debug(f"cleanup_session_images: no-op for session {session_id}")
