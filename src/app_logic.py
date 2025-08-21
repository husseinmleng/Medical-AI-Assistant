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
def validate_audio_data(audio_bytes: bytes) -> tuple[bool, str]:
    """Validates audio data before sending to transcription API."""
    if not audio_bytes:
        return False, "No audio data provided"
    
    if len(audio_bytes) < 100:
        return False, "Audio data too small (less than 100 bytes)"
    
    if len(audio_bytes) > 25 * 1024 * 1024:  # 25MB limit
        return False, "Audio file too large (over 25MB)"
    
    # Check for common audio file signatures
    audio_signatures = {
        b'\xff\xfb': "MP3",
        b'ID3': "MP3 with ID3 tag",
        b'RIFF': "WAV",
        b'\x00\x00\x00\x20ftyp': "MP4/M4A",
        b'OggS': "OGG",
        b'fLaC': "FLAC"
    }
    
    for signature, format_name in audio_signatures.items():
        if audio_bytes.startswith(signature):
            return True, f"Valid {format_name} audio detected"
    
    # If no signature found, still try but warn
    return True, "Unknown audio format (will attempt transcription)"

async def transcribe_audio(audio_bytes: bytes, lang: str) -> str:
    """Transcribes audio to text using OpenAI's Whisper model."""
    if not audio_bytes:
        return ""
    
    try:
        # Validate audio data
        is_valid, validation_msg = validate_audio_data(audio_bytes)
        if not is_valid:
            print(f"WARNING: Audio data validation failed: {validation_msg}")
            print("   - Attempting transcription despite validation warning.")
        
        audio_io = io.BytesIO(audio_bytes)
        audio_io.name = "temp_audio.wav"  # Use WAV format which is more widely supported
        
        # Try with different models and formats
        try:
            transcription_response = transcription_client.audio.transcriptions.create(
                model="whisper-1",  # Use the standard Whisper model
                file=audio_io,
                response_format="text",
                language=lang
            )
            return transcription_response
        except Exception as whisper_error:
            print(f"Whisper-1 failed, trying gpt-4o-mini: {whisper_error}")
            
            # Reset the BytesIO object
            audio_io.seek(0)
            
            try:
                # Try with gpt-4o-mini model
                transcription_response = transcription_client.audio.transcriptions.create(
                    model="gpt-4o-mini",
                    file=audio_io,
                    response_format="text",
                    language=lang
                )
                return transcription_response
            except Exception as gpt_error:
                print(f"Both models failed. Whisper-1: {whisper_error}, GPT-4o: {gpt_error}")
                
                # Try one more time with different parameters
                audio_io.seek(0)
                try:
                    transcription_response = transcription_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_io,
                        response_format="text"
                        # Don't specify language to let Whisper auto-detect
                    )
                    return transcription_response
                except Exception as final_error:
                    print(f"All transcription attempts failed: {final_error}")
                    return ""
            
    except Exception as e:
        error_msg = str(e)
        if "Audio file might be corrupted or unsupported" in error_msg:
            print("ERROR: Audio file is corrupted or in unsupported format")
            print("   - Ensure the audio file is a valid MP3, WAV, M4A, or MP4 file")
            print("   - Check if the file was properly recorded")
            print("   - Try recording again with a shorter message")
        elif "file" in error_msg.lower() and "invalid" in error_msg.lower():
            print("ERROR: Invalid audio file format")
            print("   - The audio format is not supported by OpenAI")
            print("   - Try converting to MP3 or WAV format")
        else:
            print(f"ERROR: An unexpected error occurred during audio transcription: {e}")
        
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
        print(f"=== run_graph called ===")
        print(f"Session ID: {session_id}")
        print(f"User input: {user_input[:100] if user_input else 'None'}...")
        print(f"Image path: {image_path}")
        print(f"File paths: {file_paths}")
        print(f"Language: {lang}")
        
        config = {"configurable": {"thread_id": session_id}}
        
        # Get current state for debugging
        try:
            current_state = app.get_state(config)
            print(f"Current state keys: {list(current_state.values.keys()) if current_state else 'No state'}")
            print(f"Current messages count: {len(current_state.values.get('messages', [])) if current_state else 0}")
        except Exception as e:
            print(f"Could not get current state: {e}")
        
        # Always use the user input as the human message content for consistency
        current_input = user_input
        print(f"Using user input: {current_input[:100] if current_input else 'None'}...")
        
        # Build the graph input with all necessary data
        messages = [HumanMessage(content=current_input)]
        graph_input = {"messages": messages, "lang": lang}
        
        # Add file paths to graph input - this is the primary mechanism for file uploads
        if file_paths:
            graph_input["report_file_paths"] = file_paths
            print(f"🗂️ Added report_file_paths to graph input: {len(file_paths)} files")
        
        if image_path:
            graph_input["uploaded_image_path"] = image_path
            print(f"🖼️ Added uploaded_image_path to graph input: {image_path}")
            
        # IMPORTANT: Don't pre-update state - let the graph handle it properly through routing
        
        print(f"Final graph input keys: {list(graph_input.keys())}")
        print(f"Graph input messages count: {len(graph_input['messages'])}")
        
        print("Calling app.invoke...")
        final_state = app.invoke(graph_input, config)
        print(f"App.invoke completed. Final state keys: {list(final_state.keys())}")
        
        ai_messages = [m for m in final_state.get("messages", []) if isinstance(m, AIMessage)]
        tts_audio = None
        if ai_messages:
            # Check for technical messages before generating TTS
            last_ai_message = ai_messages[-1].content
            if "INTERPRETATION_RESULT:" not in last_ai_message:
                try:
                    # Handle async context properly - check if we're in an event loop
                    import asyncio
                    try:
                        # Try to get the current loop
                        loop = asyncio.get_running_loop()
                        # We're in an async context, create task instead of run
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, text_to_speech(last_ai_message))
                            tts_audio = future.result(timeout=10)  # 10 second timeout
                    except RuntimeError:
                        # No running loop, safe to use asyncio.run
                        tts_audio = asyncio.run(text_to_speech(last_ai_message))
                except Exception as e:
                    print(f"TTS generation failed: {e}")
                    tts_audio = None  # Continue without TTS
        
        result = {
            "messages": final_state.get("messages", []),
            "annotated_image_path": final_state.get("annotated_image_path"),
            "tts_audio": tts_audio,
            "interpretation_result": final_state.get("interpretation_result"),
            "reports_text_context": final_state.get("reports_text_context")
        }
        
        print(f"Returning result with {len(result['messages'])} messages")
        return result
        
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
    app.update_state(config, empty_state)
    print(f"Cleared session history for {session_id}")

def reset_for_new_report(session_id: str, preserve_lang: str | None = None):
    """
    Clears only report-related data while preserving conversation messages, ML results, and X-ray results.
    Enhanced to maintain conversation flow while allowing new report analysis.
    """
    config = {"configurable": {"thread_id": session_id}}
    try:
        state_record = app.get_state(config)
        current_lang = preserve_lang or (state_record.values.get('lang') if state_record else None) or 'en'
        # Preserve ALL existing data except report-specific items
        existing_messages = state_record.values.get('messages', []) if state_record else []
        existing_questionnaire_inputs = state_record.values.get('questionnaire_inputs') if state_record else None
        existing_ml_result = state_record.values.get('ml_result') if state_record else None
        existing_ml_confidence = state_record.values.get('ml_confidence') if state_record else None
        existing_xray_result = state_record.values.get('xray_result') if state_record else None
        existing_xray_confidence = state_record.values.get('xray_confidence') if state_record else None
        existing_annotated_image_path = state_record.values.get('annotated_image_path') if state_record else None
    except Exception:
        current_lang = preserve_lang or 'en'
        existing_messages = []
        existing_questionnaire_inputs = None
        existing_ml_result = None
        existing_ml_confidence = None
        existing_xray_result = None
        existing_xray_confidence = None
        existing_annotated_image_path = None
    
    reset_state: GraphState = {
        "messages": existing_messages,  # PRESERVE conversation messages
        "questionnaire_inputs": existing_questionnaire_inputs,  # Preserve questionnaire data
        "ml_result": existing_ml_result,  # Preserve existing ML results
        "ml_confidence": existing_ml_confidence,  # Preserve existing ML confidence
        "xray_result": existing_xray_result,  # Preserve existing X-ray results
        "xray_confidence": existing_xray_confidence,  # Preserve existing X-ray confidence
        "annotated_image_path": existing_annotated_image_path,  # Preserve existing annotated image
        "interpretation_result": None,  # Clear only report-related data
        "report_file_paths": None,
        "uploaded_image_path": None,
        "reports_text_context": None,
        "lang": current_lang,
    }
    app.update_state(config, reset_state)
    print(f"Reset session for new report in {session_id} (lang={current_lang}, preserved {len(existing_messages)} messages, ML/X-ray results)")

def has_report_context(session_id: str) -> bool:
    """Check if the session has report interpretation context available for chat."""
    state = get_session_state(session_id)
    return bool(state.get("interpretation_result") or state.get("reports_text_context"))

def reset_for_new_xray(session_id: str, preserve_lang: str | None = None):
    """
    Clears only X-ray-related data while preserving conversation messages, ML results, and report results.
    Enhanced to maintain conversation flow while allowing new X-ray analysis.
    """
    config = {"configurable": {"thread_id": session_id}}
    try:
        state_record = app.get_state(config)
        current_lang = preserve_lang or (state_record.values.get('lang') if state_record else None) or 'en'
        # Preserve ALL existing data except X-ray-specific items
        existing_messages = state_record.values.get('messages', []) if state_record else []
        existing_questionnaire_inputs = state_record.values.get('questionnaire_inputs') if state_record else None
        existing_ml_result = state_record.values.get('ml_result') if state_record else None
        existing_ml_confidence = state_record.values.get('ml_confidence') if state_record else None
        existing_interpretation_result = state_record.values.get('interpretation_result') if state_record else None
        existing_reports_text_context = state_record.values.get('reports_text_context') if state_record else None
    except Exception:
        current_lang = preserve_lang or 'en'
        existing_messages = []
        existing_questionnaire_inputs = None
        existing_ml_result = None
        existing_ml_confidence = None
        existing_interpretation_result = None
        existing_reports_text_context = None
    
    reset_state: GraphState = {
        "messages": existing_messages,  # PRESERVE conversation messages
        "questionnaire_inputs": existing_questionnaire_inputs,  # Preserve questionnaire data
        "ml_result": existing_ml_result,  # Preserve existing ML results
        "ml_confidence": existing_ml_confidence,  # Preserve existing ML confidence
        "xray_result": None,  # Clear only X-ray related data
        "xray_confidence": None,
        "annotated_image_path": None,
        "interpretation_result": existing_interpretation_result,  # Preserve existing report results
        "report_file_paths": None,
        "uploaded_image_path": None,
        "reports_text_context": existing_reports_text_context,  # Preserve existing report context
        "lang": current_lang,
    }
    app.update_state(config, reset_state)
    print(f"Reset session for new xray in {session_id} (lang={current_lang}, preserved {len(existing_messages)} messages, ML/report results)")
