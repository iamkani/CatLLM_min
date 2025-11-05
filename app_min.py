import os
import io
import math
import re
from collections import Counter
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd

# Configure the page early before any UI is drawn
st.set_page_config(page_title="CatLLM — Minimal (Docs)", page_icon="🐾", layout="wide")

# ------------------------
# Role definitions & credentials
# ------------------------
# Each role has a dedicated system prompt. These prompts provide grounding and
# context when interacting with the OpenAI API. They emphasise grounded answers
# and encourage the use of inline citations when sources are available.
ROLE_SYSTEM_MESSAGES: Dict[str, str] = {
    "Association Analyst": (
        "You are an Association Analyst. "
        "You specialize in interpreting association records, market trends, and industry benchmarks for cattle operations. "
        "Provide data‑driven insights and analyses, and quote sources when relevant. "
        "Prefer grounded answers with inline quotes like: > snippet [source]. If unsure, say so."
    ),
    "Buyer / Feeder": (
        "You are a Buyer / Feeder. "
        "You focus on purchasing and feeding cattle, and provide guidance on feed strategies, pricing, and buying decisions. "
        "Quote sources when relevant and prefer grounded answers with inline quotes like: > snippet [source]. If unsure, say so."
    ),
    "Genetic Advisor": (
        "You are a Genetic Advisor. "
        "You specialize in cattle genetics, breeding strategies, and selection. "
        "Provide advice grounded in genetics and science, quoting sources when relevant. "
        "Prefer grounded answers with inline quotes like: > snippet [source]. If unsure, say so."
    ),
    "Independent Rancher": (
        "You are an Independent Rancher. "
        "You run a cattle ranch independently, focusing on raising cattle, pasture management, and general operations. "
        "Provide practical guidance and insights, quoting sources when relevant. "
        "Prefer grounded answers with inline quotes like: > snippet [source]. If unsure, say so."
    ),
    "Root": (
        "You are the root user with full access. "
        "Provide concise, helpful answers with inline quotes like: > snippet [source]. "
        "If unsure, say so."
    ),
}

# Accept many role spellings; normalize to keys used in ROLE_SYSTEM_MESSAGES
ROLE_ALIASES = {
    # snake_case from env → UI label keys
    "association_analyst": "Association Analyst",
    "buyer_feeder": "Buyer / Feeder",
    "genetic_advisor": "Genetic Advisor",
    "independent_rancher": "Independent Rancher",
    # fallbacks
    "root": "Root",
}

def normalize_role(role: str | None) -> str:
    if not role:
        return "Root"
    r = role.strip().lower().replace("/", "_").replace(" ", "_")
    return ROLE_ALIASES.get(r, ROLE_ALIASES.get("association_analyst"))

def parse_credentials() -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """
    Parse credentials from env:
      CATLLM_ROOT_CREDS = "admin:supersecret,ops:opspass"      (pairs)
      CATLLM_USERS      = "alexa:pw1:association_analyst,..."  (triplets)
    Returns:
      roots:  {username: password}
      users:  {username: (password, normalized_role)}
    """
    # Roots: multiple username:password pairs
    roots: Dict[str, str] = {}
    for part in filter(None, [p.strip() for p in os.getenv("CATLLM_ROOT_CREDS", "").split(",")]):
        if ":" in part:
            u, pw = part.split(":", 1)
            roots[u.strip()] = pw.strip()

    # Users: username:password:role triplets (role is required in your env)
    users: Dict[str, tuple[str, str]] = {}
    for part in filter(None, [p.strip() for p in os.getenv("CATLLM_USERS", "").split(",")]):
        bits = part.split(":")
        if len(bits) != 3:
            # ignore malformed entries
            continue
        u, pw, role = bits[0].strip(), bits[1].strip(), bits[2].strip()
        users[u] = (pw, normalize_role(role))

    # Demo defaults only if both empty
    if not roots and not users:
        roots = {"root": "root123"}
        users = {"user": ("user123", normalize_role("association_analyst"))}

    return roots, users

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
def minimal_model_reply(
    user_text: str,
    history: List[Dict],
    context_chunks: List[str],
    role_name: str = None,
) -> str:
    """Invoke OpenAI Chat Completions with role‑aware system prompts.

    If an error occurs (e.g. model not available or API key missing), a local
    heuristic is used to surface the most relevant context snippets. The
    ``role_name`` parameter determines which system prompt from
    ``ROLE_SYSTEM_MESSAGES`` is used; if not provided or unknown, the
    ``Root`` prompt is used.

    Args:
        user_text: The user's query.
        history: A list of previous messages, each with ``role`` and ``content``.
        context_chunks: A list of context strings retrieved from documents.
        role_name: The current user's role; selects the system prompt.
    Returns:
        A string reply from the model or the fallback heuristic.
    """
    # Build a preface that embeds retrieved context (if any)
    preface = ""
    if context_chunks:
        ctx_joined = "\n\n---\n".join(context_chunks[:4])
        preface = (
            "Use the following context to answer (quote relevant lines and sources):\n\n"
            + ctx_joined
            + "\n\n"
        )

    # Determine system prompt based on role
    role_canonical = normalize_role(role_name)
    system_prompt = ROLE_SYSTEM_MESSAGES.get(role_canonical, ROLE_SYSTEM_MESSAGES["Root"])
    try:
        from openai import OpenAI  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=api_key)

        # Build the message list: start with the role‑specific system prompt
        msgs = [
            {"role": "system", "content": system_prompt},
        ]
        # Keep recent conversation turns (last 6) for context
        for m in history[-6:]:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role in {"user", "assistant"} and content:
                msgs.append({"role": role, "content": content})
        # Append the current query with context preface
        msgs.append({"role": "user", "content": preface + user_text})

        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=msgs,
            temperature=0.2,
        )
        return resp.choices[0].message.content or "(no content)"
    except Exception:
        # Local fallback: surface best snippets
        if context_chunks:
            snippets = "\n\n---\n".join(context_chunks[:3])
            return (
                "(local heuristic)\nTop snippets:\n\n"
                + snippets
                + "\n\nYour question: "
                + user_text
            )
        return f"(minimal echo) {user_text}"

# ------------------------
# Simple text utils
# ------------------------
_STOPWORDS = set(
    (
        "a an and the of to in is it that this for on with as at by from be are was were or if "
        "then so such via into up out over under within without about between across not no yes you "
        "your yours we us our they their i me my mine he she him her his hers its who whom which what "
        "when where why how been being do does did done can could should would may might will shall "
        "just only also etc"
    ).split()
)

def normalize_text(s: str) -> str:
    s = s.replace("\x00", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def tokenize(s: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z0-9_]+", s.lower()) if t not in _STOPWORDS]

def chunk_text(s: str, chunk_chars: int = 1200, overlap: int = 150) -> List[str]:
    s = normalize_text(s)
    if len(s) <= chunk_chars:
        return [s]
    chunks: List[str] = []
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
    scored: List[Tuple[float, str]] = [(score_chunk(query, c), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for s, c in scored if s > 0][:k]

# ------------------------
# Extraction for uploads + source-aware chunking
# ------------------------
def extract_text_from_upload(file) -> Tuple[str, str, Dict]:
    """Return (text, info_str, meta) for an uploaded file.

    The ``meta`` dictionary may include 'type', 'pages' (list of page texts), or 'df' for
    tabular data. The ``info_str`` gives a short description of the file. Unsupported
    formats return an empty string and a descriptive message.
    """
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
            text = "\n".join(p.text for p in doc.paragraphs)
            return normalize_text(text), f"{name} (DOCX paragraphs: {len(doc.paragraphs)})", {"type": "docx"}

        if suffix in {"pdf"}:
            if not HAS_PYPDF:
                return "", f"{name} (PDF) requires 'pypdf' to extract text.", {"type": "pdf_error"}
            from pypdf import PdfReader  # type: ignore
            bio = io.BytesIO(file.read())
            reader = PdfReader(bio)
            pages: List[str] = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    pages.append("")
            text = "\n".join(pages)
            return normalize_text(text), f"{name} (PDF pages: {len(reader.pages)})", {"type": "pdf", "pages": pages}

        return "", f"{name}: unsupported file type '{suffix}'.", {"type": "unsupported"}

    except Exception as e:
        return "", f"{name}: error extracting text: {e}", {"type": "error"}


def build_source_marked_chunks(name: str, meta: Dict, text: str) -> List[str]:
    """Create source-marked text chunks for retrieval.

    For PDFs, chunk by page and add page numbers; for CSV/Excel files, chunk
    by row windows; for other types, normal text chunking with a filename marker.
    """
    t = meta.get("type", "text")
    # PDF: chunk by page for clear markers
    if t == "pdf" and meta.get("pages") is not None:
        out: List[str] = []
        for i, pg in enumerate(meta["pages"], start=1):
            for c in chunk_text(pg, chunk_chars=1200, overlap=120):
                out.append(f"[{name} | pdf | page {i}] \n{c}")
        return out
    # CSV/Excel: chunk by row windows for clear row ranges
    if t in {"csv", "excel"} and meta.get("df") is not None:
        df: pd.DataFrame = meta["df"]
        window = 40
        out: List[str] = []
        for start in range(0, len(df), window):
            end = min(len(df), start + window)
            block_csv = df.iloc[start:end].to_csv(index=False)
            for c in chunk_text(block_csv, chunk_chars=1600, overlap=0):
                out.append(f"[{name} | {t} | rows {start+1}–{end}] \n{c}")
        return out
    # DOCX/TXT/others: normal chunking with filename marker
    return [f"[{name} | {t}] \n{c}" for c in chunk_text(text, chunk_chars=1400, overlap=200)]

# ------------------------
# Session state and authentication
# ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs" not in st.session_state:
    st.session_state.docs = []
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "auth" not in st.session_state:
    # Tracks whether the user is logged in, their username, and role
    st.session_state.auth = {"is_authed": False, "username": None, "role": None}

# Parse credentials on each run. Root credentials and users are derived from env vars.
root_creds, user_creds = parse_credentials()

# Authentication: if the user is not logged in, show a simple login form
if not st.session_state.auth["is_authed"]:
    st.title("🔐 Sign In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Log in"):
        # Check root credentials (multiple roots supported)
        if username in root_creds and root_creds[username] == password:
            st.session_state.auth = {
                "is_authed": True,
                "username": username,
                "role": "Root",
            }
            st.rerun()
        # Check normal users (password, role) tuples
        elif username in user_creds and user_creds[username][0] == password:
            role_name = user_creds[username][1]
            st.session_state.auth = {
                "is_authed": True,
                "username": username,
                "role": role_name,
            }
            st.rerun()
        else:
            st.error("Invalid username or password")
    # Halt execution of the rest of the app until authenticated
    st.stop()

# ------------------------
# UI controls (top-level, not sidebar)
# ------------------------
# Indicate which user is logged in and provide a logout mechanism
st.info(f"Logged in as **{st.session_state.auth['username']}** ({st.session_state.auth['role']})")
if st.button("Log out"):
    # Reset authentication and clear session state
    st.session_state.auth = {"is_authed": False, "username": None, "role": None}
    st.session_state.messages = []
    st.session_state.docs = []
    st.session_state.all_chunks = []
    st.rerun()

# Primary controls for chat and streaming. These controls are displayed in columns.
colA, colB = st.columns([1, 6])
with colA:
    if st.button("Clear chat", key="btn_clear_chat"):
        st.session_state.messages = []
        st.rerun()
with colB:
    st.toggle("Stream responses (visual only)", value=True, key="stream_vis")

st.markdown("### 📎 Add documents")
uploads = st.file_uploader(
    "Drop files here (pdf, csv, xlsx, xls, docx, txt) — multiple allowed",
    type=["pdf", "csv", "xlsx", "xls", "docx", "txt"],
    accept_multiple_files=True,
    key="uploader_docs",
)

# A button to clear all loaded documents directly under the uploader
if st.button("Clear documents", key="btn_clear_docs_top"):
    st.session_state.docs = []
    st.session_state.all_chunks = []
    st.rerun()

# Process newly uploaded files
if uploads:
    for f in uploads:
        text, info, meta = extract_text_from_upload(f)
        if text:
            chunks = build_source_marked_chunks(f.name, meta, text)
            entry: Dict = {
                "name": f.name,
                "meta": info,
                "text": text,
                "chunks": chunks,
                "type": meta.get("type", "text"),
            }
            if meta.get("df") is not None:
                entry["df_preview"] = meta["df"].head(50)
            st.session_state.docs.append(entry)
            st.session_state.all_chunks.extend(chunks)
            st.success(f"Added: {info} (chunks: {len(chunks)})")
        else:
            st.warning(f"Skipped: {info}")

# Display loaded documents with previews and optional table previews
if st.session_state.docs:
    st.markdown("#### 📚 Loaded documents")
    for i, d in enumerate(st.session_state.docs):
        with st.expander(f"{d['name']} — {d['meta']}", expanded=False):
            if d.get("df_preview") is not None:
                st.markdown("**Table preview (first 50 rows):**")
                st.dataframe(
                    d["df_preview"],
                    use_container_width=True,
                    hide_index=True,
                    key=f"dfprev_{i}",
                )
            st.text_area(
                "Extracted text (preview)",
                d["text"][:5000],
                height=180,
                key=f"preview_{i}_{d['name']}",
            )
            st.caption(
                "Source markers are added to retrieved snippets, e.g., [filename | type | page N] or [filename | type | rows i–j]."
            )

# Optional summarization across all loaded documents
if st.session_state.docs:
    if st.button("🧠 Summarize loaded documents", key="summarize_btn"):
        context: List[str] = []
        for d in st.session_state.docs:
            sample = d["text"][:1500]
            meta_info = d["meta"]
            context.append(f"{d['name']} — {meta_info}\n{sample}")
        if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI  # type: ignore
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                prompt = (
                    "Summarize the following documents in bullet points. Keep it concise and group by filename. "
                    "Quote key lines with [source markers].\n\n"
                    + "\n\n---\n\n".join(context)
                )
                resp = client.chat.completions.create(
                    model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "You are a concise technical summarizer."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                st.markdown(resp.choices[0].message.content or "(no content)")
            except Exception as e:
                st.error(f"Model summary failed, falling back to local: {e}")
                agg = "### Local summary\n"
                for d in st.session_state.docs:
                    toks = tokenize(d["text"])
                    common_terms = ", ".join(w for w, _ in Counter(toks).most_common(10))
                    agg += f"- **{d['name']}**: {len(d['text'])} chars; top terms: {common_terms}\n"
                st.markdown(agg)
        else:
            agg = "### Local summary\n"
            for d in st.session_state.docs:
                toks = tokenize(d["text"])
                common_terms = ", ".join(w for w, _ in Counter(toks).most_common(10))
                agg += f"- **{d['name']}**: {len(d['text'])} chars; top terms: {common_terms}\n"
            st.markdown(agg)

# Separator
st.markdown("---")

# Render chat history
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input with retrieval augmentation. The role_name is passed to the model function.
prompt = st.chat_input("Ask a question about the uploaded docs—or chat normally…", key="chat_input")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chunks = st.session_state.all_chunks if st.session_state.all_chunks else []
    ctx = top_chunks(prompt, chunks, k=5) if chunks else []

    with st.chat_message("assistant"):
        with st.spinner("Analyzing…"):
            reply = minimal_model_reply(
                prompt,
                st.session_state.messages,
                ctx,
                role_name=st.session_state.auth.get("role"),
            )
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

# Footer tip
st.caption("Tip: install optional extras for richer parsing: `pypdf`, `python-docx`, `openpyxl`.")
