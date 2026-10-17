import logging
import os
from typing import Annotated, Generator, List, Optional, Tuple, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

try:
    from supabase import Client, create_client
except ImportError:
    Client = None
    create_client = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agentic_chatbot_backend")

# Load environment variables
load_dotenv()


# ---------------------------------------------------------------------------
# 1. LLM Provider Setup
# ---------------------------------------------------------------------------
def get_llm(model_provider: str = "auto"):
    """
    Initializes and returns the LLM based on available keys and preferred provider.
    Defaults to Gemini if GEMINI_API_KEY / GOOGLE_API_KEY is available.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if model_provider == "gemini" or (model_provider == "auto" and gemini_key):
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=gemini_key,
            temperature=0.7,
        )
    elif model_provider == "openai" or (model_provider == "auto" and openai_key):
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_key,
            temperature=0.7,
        )
    else:
        raise ValueError(
            "No valid API key found. Please set GEMINI_API_KEY or OPENAI_API_KEY in your .env file."
        )


llm = get_llm()


# ---------------------------------------------------------------------------
# 2. Supabase / PostgreSQL Checkpointer Initialization
# ---------------------------------------------------------------------------
def get_checkpointer() -> Tuple[BaseCheckpointSaver, str]:
    """
    Initializes a persistent checkpointer using Supabase's PostgreSQL database.
    If SUPABASE_DB_URL is not set or fails to connect, falls back to MemorySaver (RAM).
    """
    db_uri = os.environ.get("SUPABASE_DB_URL") or os.environ.get("DATABASE_URL")

    if not db_uri:
        logger.warning(
            "SUPABASE_DB_URL is not set in .env. Using in-memory checkpointer (MemorySaver). "
            "To enable persistent Supabase storage, set SUPABASE_DB_URL in .env."
        )
        return MemorySaver(), "memory"

    try:
        # Supabase transaction poolers require prepare_threshold=0
        pool = ConnectionPool(
            conninfo=db_uri,
            max_size=10,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )
        checkpointer = PostgresSaver(pool)
        # Automatically creates checkpoints, checkpoint_blobs, and checkpoint_writes tables
        checkpointer.setup()
        logger.info("Successfully connected to Supabase PostgreSQL checkpointer!")
        return checkpointer, "supabase_postgres"
    except Exception as exc:
        logger.error(
            f"Failed to connect to Supabase PostgreSQL checkpointer ({exc}). "
            "Falling back to MemorySaver."
        )
        return MemorySaver(), "memory_fallback"


checkpointer, checkpointer_type = get_checkpointer()


# ---------------------------------------------------------------------------
# 3. Supabase REST Client (Optional helper for custom tables / Auth)
# ---------------------------------------------------------------------------
def get_supabase_client() -> Optional["Client"]:
    """
    Initializes and returns the Supabase REST Client if SUPABASE_URL and SUPABASE_KEY are provided.
    Useful for custom table operations (e.g. logging user activity or managing auth).
    """
    if create_client is None:
        logger.warning("supabase-py library is not installed.")
        return None

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    if supabase_url and supabase_key:
        try:
            return create_client(supabase_url, supabase_key)
        except Exception as exc:
            logger.error(f"Error creating Supabase client: {exc}")
            return None
    return None


supabase_client = get_supabase_client()


# ---------------------------------------------------------------------------
# 4. State Definition & LangGraph Workflow
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """Core chat node that queries the model with conversation messages."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


workflow = StateGraph(ChatState)
workflow.add_node("chat_node", chat_node)
workflow.add_edge(START, "chat_node")
workflow.add_edge("chat_node", END)

# Compile chatbot graph with the active checkpointer
chatbot = workflow.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 5. Exported Functions for Frontend (Streamlit / app.py / API)
# ---------------------------------------------------------------------------
def invoke_chatbot(user_message: str, thread_id: str = "default_thread") -> str:
    """
    Invokes the chatbot synchronously with session persistence.
    The conversation is saved to Supabase (or MemorySaver) under `thread_id`.
    """
    config = {"configurable": {"thread_id": thread_id}}
    input_data = {"messages": [HumanMessage(content=user_message)]}
    result = chatbot.invoke(input_data, config=config)
    last_message = result["messages"][-1]
    return last_message.content


def stream_chatbot(
    user_message: str, thread_id: str = "default_thread"
) -> Generator[str, None, None]:
    """
    Streams the assistant response token-by-token or chunk-by-chunk.
    Compatible with Streamlit's `st.write_stream(...)`.
    """
    config = {"configurable": {"thread_id": thread_id}}
    input_data = {"messages": [HumanMessage(content=user_message)]}

    for event in chatbot.stream(input_data, config=config, stream_mode="messages"):
        msg, metadata = event
        if isinstance(msg, AIMessage) and msg.content:
            yield msg.content


def get_chat_history(thread_id: str = "default_thread") -> List[BaseMessage]:
    """
    Retrieves the entire conversation message history for a given thread_id.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)
    if state and state.values and "messages" in state.values:
        return state.values["messages"]
    return []


# ---------------------------------------------------------------------------
# 6. Direct Execution / Self-Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n--- Agentic Chatbot Backend Initialized ---")
    print(f"Checkpointer Storage Mode: {checkpointer_type.upper()}")
    print(f"Supabase REST Client: {'Configured' if supabase_client else 'Not configured (Optional)'}")
    print(f"LLM: {llm.__class__.__name__}")
    print("-------------------------------------------\n")

    test_thread = "cli-test-supabase-thread"
    print("Testing conversation flow:")
    
    reply1 = invoke_chatbot("Hello! I am planning a trip to Tokyo.", thread_id=test_thread)
    print("AI (Turn 1):", reply1)
    
    reply2 = invoke_chatbot("Where did I say I was planning to travel?", thread_id=test_thread)
    print("AI (Turn 2):", reply2)
