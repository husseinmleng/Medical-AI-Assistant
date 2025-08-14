# src/app_logic.py

import asyncio
import io
from typing import List
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
    
async def text_to_speech(text: str, voice: str = "alloy") -> bytes:
    """
    Converts text to speech using OpenAI's TTS model.
    Returns raw audio bytes that can be played or saved.
    """
    try:
        tts_client = OpenAI()
        with tts_client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text
        ) as response:
            audio_bytes = response.read()  # Read full audio
        return audio_bytes
    except Exception as e:
        print(f"ERROR: TTS generation failed: {e}")
        return b""

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
def run_graph(user_input: str, session_id: str, lang: str, image_path: str = None, file_paths: List[str] = None):
    """
    Main function to run the LangGraph-based chat.
    """
    try:
        config = {"configurable": {"thread_id": session_id}}

        if image_path:
            current_input = image_path
        elif file_paths:
            # For report interpretation, keep a simple textual user message; paths go through state
            current_input = user_input
        else:
            current_input = user_input

        messages = [HumanMessage(content=current_input)]

        graph_input = {"messages": messages, "lang": lang}
        if file_paths:
            graph_input["report_file_paths"] = file_paths
        if image_path:
            graph_input["uploaded_image_path"] = image_path

        final_state = app.invoke(graph_input, config)

        ai_messages = [m for m in final_state.get("messages", []) if isinstance(m, AIMessage)]
        tts_audio = None
        if ai_messages:
            # Check for technical messages before generating TTS
            last_ai_message = ai_messages[-1].content
            if "INTERPRETATION_RESULT:" not in last_ai_message:
                 tts_audio = asyncio.run(text_to_speech(last_ai_message))

        return {
            "messages": final_state.get("messages", []),
            "annotated_image_path": final_state.get("annotated_image_path"),
            "tts_audio": tts_audio  # New field for audio
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
        "interpretation_result": None,
        "lang": 'en' 
    }
    # Use the pre-compiled app to update the state.
    app.update_state(config, empty_state)
    print(f"Cleared session history for {session_id}")
