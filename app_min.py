import os
import streamlit as st

st.set_page_config(page_title="CatLLM — Minimal", page_icon="🐾", layout="centered")

st.title("🐾 CatLLM — Minimal App")
st.caption("No auth • No RAG • No ingest. Just a simple chat.")

with st.expander("What is this?", expanded=False):
    st.markdown(
        "This is a **minimal** version of the app for quick debugging and fast boots. "
        "It avoids importing any heavy modules or optional services. "
        "If `OPENAI_API_KEY` is set and the `openai` package is installed, it will call the API. "
        "Otherwise it will fall back to a local echo so the UI still works."
    )

def minimal_model_reply(user_text: str, history: list[dict]) -> str:
    """Try OpenAI Chat Completions; on error, fall back to echo."""
    try:
        from openai import OpenAI  # requires openai>=1.0
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=api_key)

        # Keep history short to minimize tokens
        msgs = [{"role": "system", "content": "You are a concise, helpful assistant."}]
        for m in history[-6:]:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role in {"user", "assistant"} and content:
                msgs.append({"role": role, "content": content})
        msgs.append({"role": "user", "content": user_text})

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            temperature=0.3,
        )
        return resp.choices[0].message.content or "(no content)"
    except Exception:
        return f"(minimal echo) {user_text}"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: controls
with st.sidebar:
    st.header("Settings")
    st.toggle("Stream responses (visual only)", value=True, key="stream_vis")
    if st.button("Clear chat"):
        st.session_state.messages = []
        # Streamlit 1.51 uses st.rerun()
        st.rerun()
    st.divider()
    st.markdown("**Environment checks**")
    st.write("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
    try:
        import openai  # noqa: F401
        st.write("`openai` package:", "available")
    except Exception:
        st.write("`openai` package:", "missing")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
prompt = st.chat_input("Type your message…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = minimal_model_reply(prompt, st.session_state.messages)
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

st.markdown("---")
st.caption("Tip: set `OPENAI_API_KEY` to enable model responses; otherwise I just echo back.")
