# Agentic AI Chatbot with LangGraph & Supabase

A stateful, multi-turn AI chatbot backend built with **LangGraph**, **Google Gemini** (and OpenAI), and persistent database checkpointing powered by **Supabase (PostgreSQL)**.

---

## Architecture & Features

- **LangGraph Stateful Graph**: Employs a stateful graph architecture with `ChatState` and message reducers (`add_messages`) to manage conversational context.
- **Supabase PostgreSQL Persistence**: Uses `langgraph-checkpoint-postgres` with connection pooling (`psycopg_pool`) to persist conversation threads permanently across restarts.
- **Supabase REST Chat Logging**: Every user/assistant exchange is logged to a `chat_messages` table via the Supabase REST client (see `supabase_schema.sql`).
- **Graceful In-Memory Fallback**: Automatically falls back to in-memory storage (`MemorySaver`) if Supabase is not configured, enabling zero-friction local development.
- **Multi-Model Support**: Primary support for Google Gemini (`gemini-2.5-flash`), with built-in fallback to OpenAI (`gpt-4o-mini`).
- **Two Frontends**: Streamlit (`app.py`) for rapid demos, and a FastAPI + HTML/JS web chat UI (`api.py` + `frontend/`) for a classic web-app feel.
- **Streaming Response**: Yields token-by-token chunks compatible with Streamlit (`st.write_stream`) and web interfaces.
- **Multi-Turn Session Memory**: Thread-based conversation tracking (`thread_id`), allowing separate user sessions to maintain their individual histories.
- **Safe Startup**: Importing the backend never crashes on missing API keys; a clear error is raised only when a chat call is made without configuration.

---

## Project Structure

```text
AgenticAi/
├── agentic_chatbot_backend.py   # Core LangGraph graph, LLM routing, and Supabase checkpointer
├── app.py                       # Streamlit web frontend interface
├── api.py                       # FastAPI server: JSON API + serves the web frontend
├── frontend/index.html          # Web chat UI (vanilla HTML/CSS/JS, no build step)
├── chatbot_workflow.ipynb       # Interactive prototyping notebook for graph experimentation
├── supabase_schema.sql          # One-time SQL: chat_messages table + index + RLS policies
├── tests/test_backend.py        # Unit tests (no API keys or network required)
├── .env.example                 # Environment variable template (copy to .env)
├── .env                         # Environment variables and API keys (never commit!)
├── LICENSE                      # MIT License
└── README.md                    # Project documentation
```

---

## Installation & Setup

### 1. Clone or Open the Workspace

Ensure your terminal is in the project directory:
```bash
cd c:\Users\User\Desktop\AgenticAi
```

### 2. Install Required Dependencies

Install the necessary libraries for LangGraph, Gemini, OpenAI, Streamlit, and Supabase PostgreSQL:

```bash
pip install -r requirements.txt
```

Or install them directly:
```bash
pip install langgraph langchain-core langchain-google-genai langchain-openai python-dotenv streamlit
pip install langgraph-checkpoint-postgres "psycopg[binary,pool]" supabase fastapi "uvicorn[standard]"
```

---

## Configuration (`.env`)

Copy `.env.example` to `.env` and fill in your credentials:

```env
# -----------------------------------------------
# 1. LLM API Keys (Gemini is enabled by default)
# -----------------------------------------------
GEMINI_API_KEY="your-google-gemini-api-key"
# Optional: OpenAI key if you prefer GPT models
OPENAI_API_KEY="your-openai-api-key"

# -----------------------------------------------
# 2. Supabase REST API (chat_messages logging)
# -----------------------------------------------
# Both naming styles are accepted by the backend.
SUPABASE_URL="https://[PROJECT-REF].supabase.co"
SUPABASE_KEY="your-publishable-or-service-role-key"
# NEXT_PUBLIC_SUPABASE_URL="https://[PROJECT-REF].supabase.co"
# NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY="sb_publishable_xxxx"

# -----------------------------------------------
# 3. Supabase PostgreSQL Persistence (Optional)
# -----------------------------------------------
# Find this in your Supabase Dashboard:
# Project Settings -> Database -> Connection string -> URI
SUPABASE_DB_URL="postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres"
```

Then run `supabase_schema.sql` once in the Supabase Dashboard (SQL Editor) to create the `chat_messages` table.

> **Note**: If `SUPABASE_DB_URL` is omitted, the backend will automatically use in-memory `MemorySaver()`. No setup is blocked if you just want to test locally!

---

## Usage

### 1. Test the Backend Directly

Run the backend script to verify LLM connection and multi-turn conversational memory:

```bash
python agentic_chatbot_backend.py
```

**Expected output:**
```text
--- Agentic Chatbot Backend Initialized ---
Checkpointer Storage Mode: SUPABASE_POSTGRES (or MEMORY)
LLM: ChatGoogleGenerativeAI
-------------------------------------------

Testing conversation flow:
AI (Turn 1): Hello! How can I help you today?
AI (Turn 2): Yes, I remember! You said...
```

### 2. Launch the Web Application (Streamlit)

Start the interactive chat UI:

```bash
streamlit run app.py
```

### 2b. Launch the Web Frontend (FastAPI + HTML/JS)

Prefer a classic web app? A dark-themed chat frontend ships in `frontend/`
and is served by the FastAPI server — no Node.js or build step needed:

```bash
uvicorn api:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

The same server exposes a JSON API:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/chat` | POST | Body `{"message": "...", "thread_id": "optional"}` → `{"reply", "thread_id"}` |
| `/api/history?thread_id=...` | GET | Full message history for a thread |
| `/api/health` | GET | LLM / Supabase / checkpointer status |
| `/docs` | GET | Auto-generated Swagger API docs |

### 3. Prototyping with Jupyter Notebook

To test individual LangGraph nodes or experiment with new graph states:

```bash
jupyter notebook chatbot_workflow.ipynb
```

### 4. Run the Unit Tests

The test suite needs **no API keys and no network** — it covers content
normalization, Supabase environment-variable resolution, and the safe
lazy LLM startup:

```bash
python -m unittest discover -s tests -v
```

### 5. Deploy to Streamlit Community Cloud (Optional)

1. Push this repository to GitHub and create an app at share.streamlit.io
   with `app.py` as the entry point.
2. In the app's **Settings → Secrets**, paste your keys in TOML form:
   ```toml
   GEMINI_API_KEY = "your-google-gemini-api-key"
   SUPABASE_URL = "https://[PROJECT-REF].supabase.co"
   SUPABASE_KEY = "your-publishable-or-service-role-key"
   ```
   `app.py` automatically bridges these secrets into environment variables,
   so no code changes are needed.

---

## Backend API Reference

`agentic_chatbot_backend.py` exports clean helper functions:

| Function | Description |
| :--- | :--- |
| `invoke_chatbot(user_message, thread_id)` | Executes the graph synchronously and returns the AI reply text. |
| `stream_chatbot(user_message, thread_id)` | Yields streaming text chunks token-by-token. |
| `get_chat_history(thread_id)` | Fetches the list of all messages (`HumanMessage`, `AIMessage`) stored for a session. |
| `save_chat_message(thread_id, role, content)` | Logs one message to the Supabase `chat_messages` table (no-op if not configured). |
| `fetch_chat_messages(thread_id, limit)` | Reads stored messages for a thread back from Supabase, oldest first. |
| `get_supabase_client()` | Returns an authenticated Supabase REST client instance (if configured). |
| `get_llm(model_provider)` | Returns the initialized LangChain chat model. |

---

## License

This project is open-source and available under the MIT License (see `LICENSE`).
