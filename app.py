"""
Streamlit web frontend for the Agentic AI Chatbot.

Run with:
    streamlit run app.py

Requires the backend module (agentic_chatbot_backend.py) and a valid
GEMINI_API_KEY or OPENAI_API_KEY in your .env file.
"""

import uuid

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="\U0001f916",
    layout="centered",
)

st.title("\U0001f916 Agentic AI Chatbot")
st.caption("LangGraph + Gemini/OpenAI with Supabase-backed persistent memory")


# ---------------------------------------------------------------------------
# Backend import (graceful failure if API keys are missing)
# ---------------------------------------------------------------------------
backend_error = None
try:
    from agentic_chatbot_backend import (
        get_chat_history,
        invoke_chatbot,
        stream_chatbot,
    )
except Exception as exc:  # e.g. missing API key raised at import time
    backend_error = exc

if backend_error is not None:
    st.error(
        "**Chatbot backend failed to start.**\n\n"
        f"`{backend_error}`\n\n"
        "Fix: create a `.env` file in the project folder with at least one key:\n\n"
        "```env\nGEMINI_API_KEY=your-google-gemini-api-key\n"
        "# or\nOPENAI_API_KEY=your-openai-api-key\n```"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Session state: one thread_id per browser session = one persistent conversation
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    # Pre-fill from the checkpointer so a refreshed page keeps its history
    history = get_chat_history(thread_id=st.session_state.thread_id)
    st.session_state.messages = [
        {
            "role": "user" if msg.__class__.__name__ == "HumanMessage" else "assistant",
            "content": msg.content if isinstance(msg.content, str) else str(msg.content),
        }
        for msg in history
    ]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Session")
    st.code(st.session_state.thread_id, language=None)
    if st.button("\U0001f504 New conversation", use_container_width=True):
        st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown(
        "Conversations persist across restarts when "
        "`SUPABASE_DB_URL` is configured; otherwise they live in memory "
        "for the current run only."
    )


# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------------------------
# Chat input + streaming reply
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(
                stream_chatbot(prompt, thread_id=st.session_state.thread_id)
            )
        except Exception:
            # Streaming can fail mid-flight (rate limits, network) -- retry sync
            reply = invoke_chatbot(prompt, thread_id=st.session_state.thread_id)
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
