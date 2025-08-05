# src/app_logic.py

import asyncio
import io
from openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Import the pre-compiled app with checkpointing from graph.py
from src.graph import app, GraphState

# --- Clients ---
transcription_client = OpenAI()

# --- App Utility Functions ---
async def transcribe_audio(audio_bytes: bytes, lang: str) -> str:
    """Transcribes audio to text using OpenAI's Whisper model."""
    if not audio_bytes:
        return ""
    try:
        audio_io = io.BytesIO(audio_bytes)
        audio_io.name = "temp_audio.mp3"  # Whisper API needs a file name
        
        transcription_response = transcription_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_io,
            response_format="text",
            language=lang
        )
        return transcription_response
    except Exception as e:
        print(f"ERROR: An error occurred during audio transcription: {e}")
        return ""

async def generate_chat_title(user_input: str) -> str:
    """Generates a short title for a new chat session."""
    try:
        title_prompt = f"Create a very short, concise title (4 words max) for a medical chat conversation that starts with this user message: '{user_input}'. Do not use quotes in the title."
        title_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        response = await title_llm.ainvoke([HumanMessage(content=title_prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"Error generating title: {e}")
        return "New Chat"

# --- Main Graph Invocation Function ---
def run_graph(user_input: str, session_id: str, lang: str, image_path: str = None):
    """
    Main function to run the LangGraph-based chat.
    """
    try:
        # Configuration for the graph run
        config = {"configurable": {"thread_id": session_id}}
        
        # Prepare the input for the graph
        current_input = image_path if image_path else user_input
        messages = [HumanMessage(content=current_input)]
        # The 'lang' key is now part of the GraphState and must be included
        graph_input = {"messages": messages, "lang": lang}

        # Invoke the graph. It already has the checkpointer from compilation.
        final_state = app.invoke(graph_input, config)
        
        return {
            "messages": final_state.get("messages", []),
            "annotated_image_path": final_state.get("annotated_image_path")
        }

    except Exception as e:
        error_msg = f"An unexpected error occurred in the graph logic: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "messages": [
                HumanMessage(content=user_input),
                AIMessage(content="I'm sorry, a system error occurred. Please try starting a new chat.")
            ],
            "annotated_image_path": None
        }

def get_history(session_id: str):
    """Retrieves conversation history using the high-level app.get_state method."""
    config = {"configurable": {"thread_id": session_id}}
    # The app already has the checkpointer, so we can call get_state directly.
    state_record = app.get_state(config)
    return state_record.values.get('messages', []) if state_record else []

def clear_session_history(session_id: str):
    """
    Clears the history for a given session ID using the high-level app.update_state.
    """
    config = {"configurable": {"thread_id": session_id}}
    # Reset the state by providing a full, empty GraphState dictionary.
    # A default 'lang' is needed to satisfy the type hint.
    empty_state = {
        "messages": [],
        "questionnaire_inputs": None,
        "ml_result": None,
        "ml_confidence": None,
        "xray_result": None,
        "xray_confidence": None,
        "annotated_image_path": None,
        "lang": 'en' 
    }
    # Use the pre-compiled app to update the state.
    app.update_state(config, empty_state)
    print(f"Cleared session history for {session_id}")
