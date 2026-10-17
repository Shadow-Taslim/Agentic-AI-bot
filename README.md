# Agentic AI Chatbot with LangGraph & Supabase

A stateful, multi-turn AI chatbot backend built with **LangGraph**, **Google Gemini** (and OpenAI), and persistent database checkpointing powered by **Supabase (PostgreSQL)**.

---

## Architecture & Features

- **LangGraph Stateful Graph**: Employs a stateful graph architecture with `ChatState` and message reducers (`add_messages`) to manage conversational context.
- **Supabase PostgreSQL Persistence**: Uses `langgraph-checkpoint-postgres` with connection pooling (`psycopg_pool`) to persist conversation threads permanently across restarts.
- **Graceful In-Memory Fallback**: Automatically falls back to in-memory storage (`MemorySaver`) if Supabase is not configured, enabling zero-friction local development.
- **Multi-Model Support**: Primary support for Google Gemini (`gemini-2.5-flash`), with built-in fallback to OpenAI (`gpt-4o-mini`).
- **Streaming Response**: Yields token-by-token chunks compatible with Streamlit (`st.write_stream`) and web interfaces.
- **Multi-Turn Session Memory**: Thread-based conversation tracking (`thread_id`), allowing separate user sessions to maintain their individual histories.

---

## Project Structure

```text
AgenticAi/
├── agentic_chatbot_backend.py   # Core LangGraph graph, LLM routing, and Supabase checkpointer
├── app.py                       # Streamlit web frontend interface
├── chatbot_workflow.ipynb       # Interactive prototyping notebook for graph experimentation
├── .env                         # Environment variables and API keys
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
pip install langgraph-checkpoint-postgres "psycopg[binary,pool]" supabase
```

---

## Configuration (`.env`)

Create or update your `.env` file in the project root with your credentials:

```env
# -----------------------------------------------
# 1. LLM API Keys (Gemini is enabled by default)
# -----------------------------------------------
GEMINI_API_KEY="your-google-gemini-api-key"
# Optional: OpenAI key if you prefer GPT models
OPENAI_API_KEY="your-openai-api-key"

# -----------------------------------------------
# 2. Supabase PostgreSQL Persistence (Optional)
# -----------------------------------------------
# Find this in your Supabase Dashboard:
# Project Settings -> Database -> Connection string -> URI
SUPABASE_DB_URL="postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres"

# Optional: Supabase REST API credentials (for auth or custom tables)
SUPABASE_URL="https://[PROJECT-REF].supabase.co"
SUPABASE_KEY="your-supabase-anon-or-service-key"
```

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

### 3. Prototyping with Jupyter Notebook

To test individual LangGraph nodes or experiment with new graph states:

```bash
jupyter notebook chatbot_workflow.ipynb
```

---

## Backend API Reference

[agentic_chatbot_backend.py](file:///c:/Users/User/Desktop/AgenticAi/agentic_chatbot_backend.py) exports clean helper functions:

| Function | Description |
| :--- | :--- |
| `invoke_chatbot(user_message, thread_id)` | Executes the graph synchronously and returns the AI reply text. |
| `stream_chatbot(user_message, thread_id)` | Yields streaming text chunks token-by-token. |
| `get_chat_history(thread_id)` | Fetches the list of all messages (`HumanMessage`, `AIMessage`) stored for a session. |
| `get_supabase_client()` | Returns an authenticated Supabase REST client instance (if configured). |
| `get_llm(model_provider)` | Returns the initialized LangChain chat model. |

---

## License

This project is open-source and available under the MIT License.