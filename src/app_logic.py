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
    Enhanced to support ongoing conversations with report results.
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
            "tts_audio": tts_audio,
            "interpretation_result": final_state.get("interpretation_result"),  # Added this
            "reports_text_context": final_state.get("reports_text_context")  # Added this
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
            "annotated_image_path": None,
            "interpretation_result": None,
            "reports_text_context": None
        }

def get_history(session_id: str):
    """Retrieves conversation history using the high-level app.get_state method."""
    config = {"configurable": {"thread_id": session_id}}
    # The app already has the checkpointer, so we can call get_state directly.
    state_record = app.get_state(config)
    return state_record.values.get('messages', []) if state_record else []

def get_session_state(session_id: str):
    """Retrieves the full session state for checking report context."""
    config = {"configurable": {"thread_id": session_id}}
    state_record = app.get_state(config)
    return state_record.values if state_record else {}

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
        "report_file_paths": None,
        "uploaded_image_path": None,
        "reports_text_context": None,
        "lang": 'en' 
    }
    # Use the pre-compiled app to update the state.
    app.update_state(config, empty_state)
    print(f"Cleared session history for {session_id}")

def reset_for_new_report(session_id: str, preserve_lang: str | None = None):
    """
    Clears prior chat/messages and any previous report interpretation/context while preserving language.
    Use this before processing a NEW report upload so that follow-up chat ties only to the latest report.
    """
    config = {"configurable": {"thread_id": session_id}}
    # Determine language to keep
    try:
        state_record = app.get_state(config)
        current_lang = preserve_lang or (state_record.values.get('lang') if state_record else None) or 'en'
    except Exception:
        current_lang = preserve_lang or 'en'

    reset_state: GraphState = {
        "messages": [],
        "questionnaire_inputs": None,
        "ml_result": None,
        "ml_confidence": None,
        "xray_result": None,
        "xray_confidence": None,
        "annotated_image_path": None,
        "interpretation_result": None,
        "report_file_paths": None,
        "uploaded_image_path": None,
        "reports_text_context": None,
        "lang": current_lang,
    }
    app.update_state(config, reset_state)
    print(f"Reset session for new report in {session_id} (lang={current_lang})")

def has_report_context(session_id: str) -> bool:
    """Check if the session has report interpretation context available for chat."""
    state = get_session_state(session_id)
    return bool(state.get("interpretation_result") or state.get("reports_text_context"))

def reset_for_new_xray(session_id: str, preserve_lang: str | None = None):
    """
    Clears prior chat/messages and any previous xray interpretation/context while preserving language.
    Use this before processing a NEW xray upload so that follow-up chat ties only to the latest xray.
    """
    config = {"configurable": {"thread_id": session_id}}
    # Determine language to keep
    try:
        state_record = app.get_state(config)
        current_lang = preserve_lang or (state_record.values.get('lang') if state_record else None) or 'en'
    except Exception:
        current_lang = preserve_lang or 'en'

    reset_state: GraphState = {
        "messages": [],
        "questionnaire_inputs": None,
        "ml_result": None,
        "ml_confidence": None,
        "xray_result": None,
        "xray_confidence": None,
        "annotated_image_path": None,
        "interpretation_result": None,
        "report_file_paths": None,
        "uploaded_image_path": None,
        "reports_text_context": None,
        "lang": current_lang,
    }
    app.update_state(config, reset_state)
    print(f"Reset session for new xray in {session_id} (lang={current_lang})")
