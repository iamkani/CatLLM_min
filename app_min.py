
import os
import io
import math
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional

import streamlit as st
import pandas as pd

st.set_page_config(page_title="CatLLM — Minimal (Docs)", page_icon="🐾", layout="wide")

st.title("🐾 CatLLM — Minimal App + Document Chat")
st.caption("Upload PDFs/CSVs/Excel/DOCX/TXT and chat with a lightweight retriever. No RAG infra required.")

# ------------------------
# Optional deps detection
# ------------------------
def has_pkg(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False

HAS_OPENAI = has_pkg("openai")
HAS_PYPDF = has_pkg("pypdf")
HAS_DOCX = has_pkg("docx")  # python-docx
HAS_OPENPYXL = has_pkg("openpyxl")
HAS_XLRD = has_pkg("xlrd")

# ------------------------
# Minimal model call
# ------------------------
def minimal_model_reply(user_text: str, history: List[Dict], context_chunks: List[str]) -> str:
    """Try OpenAI Chat Completions; on error, fall back to local heuristic answer with quoted snippets."""
    preface = ""
    if context_chunks:
        ctx_joined = "\n\n---\n".join(context_chunks[:4])
        preface = f"Use the following context to answer (quote relevant lines and sources):\n\n{ctx_joined}\n\n"

    try:
        from openai import OpenAI  # requires openai>=1.0
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=api_key)

        msgs = [
            {"role": "system", "content": "You are a concise, helpful assistant. Prefer grounded answers with inline quotes like: > snippet [source]. If unsure, say so."},
        ]
        # keep recent turns
        for m in history[-6:]:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role in {"user", "assistant"} and content:
                msgs.append({"role": role, "content": content})
        msgs.append({"role": "user", "content": preface + user_text})

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            temperature=0.2,
        )
        return resp.choices[0].message.content or "(no content)"
    except Exception:
        # Local fallback: surface best snippets
        if context_chunks:
            snippets = "\n\n---\n".join(context_chunks[:3])
            return f"(local heuristic)\nTop snippets:\n\n{snippets}\n\nYour question: {user_text}"
        return f"(minimal echo) {user_text}"

# ------------------------
# Simple text utils
# ------------------------
_STOPWORDS = set(("a an and the of to in is it that this for on with as at by from be are was were or if then so such via into up out over under within without about between across not no yes you your yours we us our they their i me my mine he she him her his hers its who whom which what when where why how been being do does did done can could should would may might will shall just only also etc").split())

def normalize_text(s: str) -> str:
    s = s.replace('\\x00', '')
    s = re.sub(r'\\s+', ' ', s)
    return s.strip()

def tokenize(s: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z0-9_]+", s.lower()) if t not in _STOPWORDS]

def chunk_text(s: str, chunk_chars: int = 1200, overlap: int = 150) -> List[str]:
    s = normalize_text(s)
    if len(s) <= chunk_chars:
        return [s]
    chunks = []
    start = 0
    while start < len(s):
        end = min(len(s), start + chunk_chars)
        chunks.append(s[start:end])
        if end == len(s):
            break
        start = max(0, end - overlap)
    return chunks

def score_chunk(query: str, chunk: str) -> float:
    # Simple bag-of-words tf scoring
    q_tokens = tokenize(query)
    if not q_tokens: 
        return 0.0
    c_counts = Counter(tokenize(chunk))
    return sum(c_counts.get(t, 0) for t in set(q_tokens)) / (1 + math.log10(1 + len(chunk)))

def top_chunks(query: str, chunks: List[str], k: int = 5) -> List[str]:
    scored = [(score_chunk(query, c), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for s, c in scored if s > 0][:k]

# ------------------------
# Extraction for uploads + source-aware chunking
# ------------------------
def extract_text_from_upload(file) -> Tuple[str, str, Dict]:
    """Return (text, info_str, meta). Meta may include 'type', 'pages' (list of page texts), or 'df'."""
    name = file.name
    suffix = name.split(".")[-1].lower()

    try:
        if suffix in {"txt", "md", "log"}:
            data = file.read().decode("utf-8", errors="ignore")
            return normalize_text(data), f"{name} (plain text)", {"type": "text"}

        if suffix in {"csv"}:
            df = pd.read_csv(file)
            preview = df.to_csv(index=False)
            return normalize_text(preview), f"{name} (CSV {df.shape[0]} rows × {df.shape[1]} cols)", {"type": "csv", "df": df}

        if suffix in {"xlsx", "xls"}:
            try:
                df = pd.read_excel(file)  # engine auto
                preview = df.to_csv(index=False)
                return normalize_text(preview), f"{name} (Excel {df.shape[0]} rows × {df.shape[1]} cols)", {"type": "excel", "df": df}
            except Exception as e:
                return "", f"{name} (Excel) could not be parsed: {e}. Install openpyxl/xlrd.", {"type": "excel_error"}

        if suffix in {"docx"}:
            if not HAS_DOCX:
                return "", f"{name} (DOCX) requires 'python-docx' to extract text.", {"type": "docx_error"}
            from docx import Document  # type: ignore
            bio = io.BytesIO(file.read())
            doc = Document(bio)
            text = "\\n".join(p.text for p in doc.paragraphs)
            return normalize_text(text), f"{name} (DOCX paragraphs: {len(doc.paragraphs)})", {"type": "docx"}

        if suffix in {"pdf"}:
            if not HAS_PYPDF:
                return "", f"{name} (PDF) requires 'pypdf' to extract text.", {"type": "pdf_error"}
            from pypdf import PdfReader  # type: ignore
            bio = io.BytesIO(file.read())
            reader = PdfReader(bio)
            pages = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    pages.append("")
            text = "\\n".join(pages)
            return normalize_text(text), f"{name} (PDF pages: {len(reader.pages)})", {"type": "pdf", "pages": pages}

        return "", f"{name}: unsupported file type '{suffix}'.", {"type": "unsupported"}

    except Exception as e:
        return "", f"{name}: error extracting text: {e}", {"type": "error"}

def build_source_marked_chunks(name: str, meta: Dict, text: str) -> List[str]:
    t = meta.get("type", "text")
    # PDF: chunk by page for clear markers
    if t == "pdf" and meta.get("pages") is not None:
        out = []
        for i, pg in enumerate(meta["pages"], start=1):
            for c in chunk_text(pg, chunk_chars=1200, overlap=120):
                out.append(f"[{name} | pdf | page {i}] \n{c}")
        return out
    # CSV/Excel: chunk by row windows for clear row ranges
    if t in {"csv", "excel"} and meta.get("df") is not None:
        df: pd.DataFrame = meta["df"]
        # windows of rows
        window = 40
        out = []
        for start in range(0, len(df), window):
            end = min(len(df), start + window)
            block_csv = df.iloc[start:end].to_csv(index=False)
            for c in chunk_text(block_csv, chunk_chars=1600, overlap=0):
                out.append(f"[{name} | {t} | rows {start+1}–{end}] \n{c}")
        return out
    # DOCX/TXT/others: normal chunking with filename marker
    return [f"[{name} | {t}] \n{c}" for c in chunk_text(text, chunk_chars=1400, overlap=200)]

# ------------------------
# Session state
# ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs" not in st.session_state:
    # each: {name, meta, text, chunks, type, preview_df?}
    st.session_state.docs = []
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []

# ------------------------
# Sidebar controls
# ------------------------
with st.sidebar:
    st.header("Settings")
    st.toggle("Stream responses (visual only)", value=True, key="stream_vis")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Clear chat", key="btn_clear_chat"):
            st.session_state.messages = []
            st.rerun()
    with colB:
        if st.button("Clear documents", key="btn_clear_docs"):
            st.session_state.docs = []
            st.session_state.all_chunks = []
            st.rerun()

    st.divider()
    st.header("Environment")
    st.write("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
    st.write("`openai` package:", "available" if HAS_OPENAI else "missing")
    st.write("PDF text (`pypdf`):", "available" if HAS_PYPDF else "missing")
    st.write("DOCX text (`python-docx`):", "available" if HAS_DOCX else "missing")
    st.write("Excel engines:", ("openpyxl " if HAS_OPENPYXL else "") + ("xlrd" if HAS_XLRD else "") or "none")

st.markdown("### 📎 Add documents")
uploads = st.file_uploader(
    "Drop files here (pdf, csv, xlsx, xls, docx, txt) — multiple allowed",
    type=["pdf", "csv", "xlsx", "xls", "docx", "txt"],
    accept_multiple_files=True,
    key="uploader_docs"
)
if uploads:
    for f in uploads:
        text, info, meta = extract_text_from_upload(f)
        if text:
            chunks = build_source_marked_chunks(f.name, meta, text)
            entry = {"name": f.name, "meta": info, "text": text, "chunks": chunks, "type": meta.get("type", "text")}
            # Store small preview dataframe for CSV/Excel
            if meta.get("df") is not None:
                entry["df_preview"] = meta["df"].head(50)  # keep small
            st.session_state.docs.append(entry)
            st.session_state.all_chunks.extend(chunks)
            st.success(f"Added: {info} (chunks: {len(chunks)})")
        else:
            st.warning(f"Skipped: {info}")

# Show current docs with previews
if st.session_state.docs:
    st.markdown("#### 📚 Loaded documents")
    for i, d in enumerate(st.session_state.docs):
        with st.expander(f"{d['name']} — {d['meta']}", expanded=False):
            # For tabular, show table preview
            if d.get("df_preview") is not None:
                st.markdown("**Table preview (first 50 rows):**")
                st.dataframe(d["df_preview"], use_container_width=True, hide_index=True, key=f"dfprev_{i}")
            # Always show text preview
            st.text_area("Extracted text (preview)", d["text"][:5000], height=180, key=f"preview_{i}_{d['name']}")
            st.caption("Source markers are added to retrieved snippets, e.g., [filename | type | page N] or [filename | type | rows i–j].")

# Summarize all docs (quick, token-safe)
if st.session_state.docs:
    if st.button("🧠 Summarize loaded documents", key="summarize_btn"):
        context = []
        for d in st.session_state.docs:
            sample = d["text"][:1500]
            meta = d["meta"]
            context.append(f"{d['name']} — {meta}\\n{sample}")
        # Use model if available, else local word frequency
        if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                prompt = "Summarize the following documents in bullet points. Keep it concise and group by filename. Quote key lines with [source markers].\\n\\n" + "\\n\\n---\\n\\n".join(context)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a concise technical summarizer."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                st.markdown(resp.choices[0].message.content or "(no content)")
            except Exception as e:
                st.error(f"Model summary failed, falling back to local: {e}")
                # Fall back
                agg = "### Local summary\\n"
                for d in st.session_state.docs:
                    toks = tokenize(d["text"])
                    common = ", ".join(w for w, _ in Counter(toks).most_common(10))
                    agg += f"- **{d['name']}**: {len(d['text'])} chars; top terms: {common}\\n"
                st.markdown(agg)
        else:
            agg = "### Local summary\\n"
            for d in st.session_state.docs:
                toks = tokenize(d["text"])
                common = ", ".join(w for w, _ in Counter(toks).most_common(10))
                agg += f"- **{d['name']}**: {len(d['text'])} chars; top terms: {common}\\n"
            st.markdown(agg)

st.markdown("---")

# Render chat history
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input with lightweight retrieval augmentation
prompt = st.chat_input("Ask a question about the uploaded docs—or chat normally…", key="chat_input")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve top chunks
    chunks = st.session_state.all_chunks if st.session_state.all_chunks else []
    ctx = top_chunks(prompt, chunks, k=5) if chunks else []

    with st.chat_message("assistant"):
        with st.spinner("Analyzing…"):
            reply = minimal_model_reply(prompt, st.session_state.messages, ctx)
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

st.caption("Tip: install optional extras for richer parsing: `pypdf`, `python-docx`, `openpyxl`.")
