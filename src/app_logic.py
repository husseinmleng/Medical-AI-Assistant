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
            "interpretation_result": final_state.get("interpretation_result"),
            "reports_text_context": final_state.get("reports_text_context")
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
    Clears prior chat/messages and any previous report interpretation/context while preserving language.
    """
    config = {"configurable": {"thread_id": session_id}}
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
    """
    config = {"configurable": {"thread_id": session_id}}
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

async def generate_and_download_report(session_id: str):
    """Generates and downloads a PDF report of the conversation."""
    from src.latex_agent import generate_latex_report
    from src.tools import convert_latex_to_pdf, check_latex_installation
    from langchain_core.messages import AIMessage, HumanMessage

    print(f"Starting PDF report generation for session: {session_id}")
    
    # First check if LaTeX is installed
    latex_status = check_latex_installation()
    if not latex_status["installed"]:
        error_msg = f"LaTeX is not properly installed. {latex_status.get('installation_guide', 'Please install a LaTeX distribution.')}"
        print(error_msg)
        return None
    
    print(f"LaTeX installation verified: {latex_status.get('version', 'Unknown version')}")
    
    state = get_session_state(session_id)
    if not state:
        print("Error: Could not retrieve session state.")
        return None

    conversation_for_report = []
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            conversation_for_report.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            conversation_for_report.append({"role": "assistant", "content": msg.content})

    # Translate the conversation if the language is Arabic
    if state.get("lang") == "ar":
        print("Translating conversation to English for the report...")
        print(f"Original conversation has {len(conversation_for_report)} messages")
        
        # Show first few messages before translation
        for i, msg in enumerate(conversation_for_report[:3]):
            print(f"Before translation - Message {i+1}: {msg.get('content', '')[:100]}...")
        
        conversation_for_report = await translate_conversation_to_english(conversation_for_report)
        
        print(f"After translation, conversation has {len(conversation_for_report)} messages")
        
        # Show first few messages after translation
        for i, msg in enumerate(conversation_for_report[:3]):
            print(f"After translation - Message {i+1}: {msg.get('content', '')[:100]}...")
        
        # Verify that translation actually happened
        if conversation_for_report and len(conversation_for_report) > 0:
            first_msg_content = conversation_for_report[0].get('content', '')
            if first_msg_content and any('\u0600' <= char <= '\u06FF' for char in first_msg_content):
                print("⚠️ WARNING: Translation may have failed - Arabic characters still present!")
                print("   This could cause LaTeX compilation issues.")
            else:
                print("✓ Translation appears successful - no Arabic characters detected")
        else:
            print("⚠️ WARNING: Translation returned empty conversation!")
    else:
        print(f"Language is {state.get('lang')}, no translation needed")

    questionnaire_data = state.get("questionnaire_inputs") or {}
    patient_name = questionnaire_data.get("patient_name", "Patient Name Not Provided")
    
    print(f"Final conversation for LaTeX generation has {len(conversation_for_report)} messages")
    print("Generating LaTeX string...")
    try:
        latex_string = generate_latex_report(
            conversation=conversation_for_report,
            patient_info=questionnaire_data,
            analysis_results={
                "ml_result": state.get("ml_result"),
                "ml_confidence": state.get("ml_confidence"),
                "xray_result": state.get("xray_result"),
                "xray_confidence": state.get("xray_confidence"),
                "annotated_image_path": state.get("annotated_image_path"),
                "interpretation_result": state.get("interpretation_result"),
                "reports_text_context": state.get("reports_text_context"),
            },
            patient_name=patient_name,
            report_title="Medical Analysis Report"
        )
        print(f"LaTeX string generated successfully (length: {len(latex_string)} characters)")
        
        # Save LaTeX string to debug file
        try:
            import os
            from datetime import datetime
            
            # Create debug directory if it doesn't exist
            debug_dir = "debug_outputs/latex"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"latex_report_{timestamp}.tex"
            filepath = os.path.join(debug_dir, filename)
            
            # Save LaTeX content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(latex_string)
            
            print(f"✓ LaTeX debug file saved: {filepath}")
            
            # Also save a metadata file
            meta_filename = f"latex_metadata_{timestamp}.txt"
            meta_filepath = os.path.join(debug_dir, meta_filename)
            
            with open(meta_filepath, 'w', encoding='utf-8') as f:
                f.write("=== LATEX GENERATION METADATA ===\n\n")
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Patient Name: {patient_name}\n")
                f.write(f"Language: {state.get('lang', 'unknown')}\n")
                f.write(f"Conversation messages: {len(conversation_for_report)}\n")
                f.write(f"LaTeX string length: {len(latex_string)} characters\n")
                f.write(f"Generation timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Report title: Medical Analysis Report\n\n")
                
                f.write("=== CONVERSATION SUMMARY ===\n")
                for i, msg in enumerate(conversation_for_report):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')[:100] + "..." if len(msg.get('content', '')) > 100 else msg.get('content', '')
                    f.write(f"Message {i+1} ({role}): {content}\n")
                
                f.write("\n=== ANALYSIS RESULTS ===\n")
                f.write(f"ML Result: {state.get('ml_result', 'N/A')}\n")
                f.write(f"ML Confidence: {state.get('ml_confidence', 'N/A')}\n")
                f.write(f"X-ray Result: {state.get('xray_result', 'N/A')}\n")
                f.write(f"X-ray Confidence: {state.get('xray_confidence', 'N/A')}\n")
                f.write(f"Annotated Image: {state.get('annotated_image_path', 'N/A')}\n")
                f.write(f"Interpretation: {state.get('interpretation_result', 'N/A')}\n")
                f.write(f"Reports Context: {state.get('reports_text_context', 'N/A')}\n")
            
            print(f"✓ LaTeX metadata file saved: {meta_filepath}")
            
        except Exception as e:
            print(f"Warning: Could not save LaTeX debug files: {e}")
        
    except Exception as e:
        print(f"Error generating LaTeX string: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print("Converting LaTeX string to PDF...")
    output_filename = f"report_{session_id}.pdf"
    pdf_path = convert_latex_to_pdf(latex_string, output_filename)
    
    if not pdf_path or "Error" in pdf_path:
        print(f"PDF generation failed: {pdf_path}")
        
        # Save error information to debug file
        try:
            import os
            from datetime import datetime
            
            debug_dir = "debug_outputs/reports"
            os.makedirs(debug_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pdf_error_{timestamp}.txt"
            filepath = os.path.join(debug_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=== PDF GENERATION ERROR ===\n\n")
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Error: {pdf_path}\n")
                f.write(f"LaTeX string length: {len(latex_string)} characters\n")
            
            print(f"✓ Error debug file saved: {filepath}")
            
        except Exception as e:
            print(f"Warning: Could not save error debug file: {e}")
        
        return None
    
    print(f"PDF generated successfully at: {pdf_path}")
    
    # Save successful PDF generation info
    try:
        import os
        from datetime import datetime
        
        debug_dir = "debug_outputs/reports"
        os.makedirs(debug_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pdf_success_{timestamp}.txt"
        filepath = os.path.join(debug_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=== PDF GENERATION SUCCESS ===\n\n")
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"PDF Path: {pdf_path}\n")
            f.write(f"PDF Filename: {output_filename}\n")
            f.write(f"LaTeX string length: {len(latex_string)} characters\n")
            f.write(f"Patient Name: {patient_name}\n")
            f.write(f"Language: {state.get('lang', 'unknown')}\n")
            f.write(f"Conversation messages: {len(conversation_for_report)}\n")
        
        print(f"✓ Success debug file saved: {filepath}")
        
    except Exception as e:
        print(f"Warning: Could not save success debug file: {e}")
    
    return pdf_path


async def translate_conversation_to_english(conversation: List[dict]) -> List[dict]:
    """
    Enhanced translation function that translates a conversation from Arabic to English using an LLM.
    Handles medical terminology and maintains conversation structure.
    """
    try:
        if not conversation:
            print("Translation: No conversation to translate")
            return conversation
        
        print(f"Translation: Starting with {len(conversation)} messages")
        
        # Check if conversation is already in English
        sample_text = " ".join([msg.get('content', '')[:100] for msg in conversation[:3]])
        if _is_likely_english(sample_text):
            print("Translation: Conversation appears to already be in English, skipping translation.")
            return conversation
        
        print("Translation: Conversation appears to be in Arabic, proceeding with translation")
        
        # Create a structured transcript for translation
        transcript_parts = []
        for i, msg in enumerate(conversation):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '').strip()
            if content:
                # Ensure content is properly encoded
                try:
                    if isinstance(content, bytes):
                        content = content.decode('utf-8', errors='replace')
                    content = str(content)
                except Exception as e:
                    print(f"Translation: Warning - Could not process message content: {e}")
                    content = str(content) if content else ""
                
                transcript_parts.append(f"[{role.upper()}_{i+1}]: {content}")
        
        if not transcript_parts:
            print("Translation: No transcript parts created")
            return conversation
        
        transcript = "\n\n".join(transcript_parts)
        print(f"Translation: Created transcript with {len(transcript_parts)} parts")
        
        # Simplified prompt to reduce complexity and improve speed
        prompt = f"""Translate this medical conversation from Arabic to English. Keep the format [USER_X] and [ASSISTANT_X].

{transcript}

Translation:"""
        
        print(f"Translation: Starting translation of {len(conversation)} messages...")
        
        # Use a faster model and add timeout
        translation_llm = ChatOpenAI(
            model="gpt-4o-mini",  # Faster than gpt-4o
            temperature=0, 
            max_tokens=2000,  # Reduced from 4000
            timeout=30  # 30 second timeout
        )
        
        # Add timeout wrapper
        import asyncio
        try:
            response = await asyncio.wait_for(
                translation_llm.ainvoke([HumanMessage(content=prompt)]),
                timeout=45  # 45 second total timeout
            )
            print("Translation: LLM response received successfully")
        except asyncio.TimeoutError:
            print("Translation: Translation timed out, using original conversation")
            return conversation
        except Exception as e:
            print(f"Translation: Translation failed: {e}, using original conversation")
            return conversation
        
        print("Translation: Translation completed, parsing response...")
        
        # Parse the translated response back into the conversation format
        translated_conversation = []
        translated_text = response.content.strip()
        print(f"Translation: Raw response length: {len(translated_text)} characters")
        
        # Extract translated messages using regex pattern
        import re
        pattern = r'\[(USER|ASSISTANT)_\d+\]:\s*(.*?)(?=\n\n\[(?:USER|ASSISTANT)_\d+\]:|$)'
        matches = re.findall(pattern, translated_text, re.DOTALL)
        print(f"Translation: Regex found {len(matches)} matches")
        
        if matches:
            for role, content in matches:
                clean_content = content.strip()
                if clean_content:
                    translated_conversation.append({
                        "role": role.lower(), 
                        "content": clean_content
                    })
            print(f"Translation: Successfully parsed {len(translated_conversation)} messages from regex")
        else:
            print("Translation: Regex parsing failed, trying line-by-line parsing")
            # Fallback: try line-by-line parsing
            lines = translated_text.split('\n')
            current_msg = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('[USER_') or line.startswith('[ASSISTANT_'):
                    if current_msg and current_msg['content']:
                        translated_conversation.append(current_msg)
                    
                    # Extract role and content
                    if ']:' in line:
                        role_part, content = line.split(']:', 1)
                        role = 'user' if 'USER' in role_part else 'assistant'
                        current_msg = {"role": role, "content": content.strip()}
                elif current_msg and line:
                    # Continue previous message
                    current_msg['content'] += ' ' + line
            
            # Don't forget the last message
            if current_msg and current_msg['content']:
                translated_conversation.append(current_msg)
            
            print(f"Translation: Line-by-line parsing found {len(translated_conversation)} messages")
        
        # Validate translation success
        if len(translated_conversation) == 0:
            print("Translation: Translation parsing failed, using original conversation")
            return conversation
        
        print(f"Translation: Successfully translated {len(conversation)} messages to English")
        
        # Save translated conversation to debug file
        try:
            import os
            from datetime import datetime
            
            # Create debug directory if it doesn't exist
            debug_dir = "debug_outputs/translations"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"translation_{timestamp}.txt"
            filepath = os.path.join(debug_dir, filename)
            
            # Save original Arabic and translated English for comparison
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=== ORIGINAL ARABIC CONVERSATION ===\n\n")
                for i, msg in enumerate(conversation):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    f.write(f"[{role.upper()}_{i+1}]: {content}\n\n")
                
                f.write("\n" + "="*50 + "\n\n")
                f.write("=== TRANSLATED ENGLISH CONVERSATION ===\n\n")
                for i, msg in enumerate(translated_conversation):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    f.write(f"[{role.upper()}_{i+1}]: {content}\n\n")
                
                f.write("\n" + "="*50 + "\n\n")
                f.write("=== TRANSLATION METADATA ===\n")
                f.write(f"Original messages: {len(conversation)}\n")
                f.write(f"Translated messages: {len(translated_conversation)}\n")
                f.write(f"Translation timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Model used: gpt-4o-mini\n")
                f.write(f"Prompt length: {len(prompt)} characters\n")
                f.write(f"Response length: {len(translated_text)} characters\n")
                f.write(f"Raw response: {translated_text}\n")
            
            print(f"Translation: ✓ Translation debug file saved: {filepath}")
            
        except Exception as e:
            print(f"Translation: Warning - Could not save translation debug file: {e}")
        
        print(f"Translation: Returning {len(translated_conversation)} translated messages")
        return translated_conversation
        
    except Exception as e:
        print(f"Translation: Error during translation: {e}")
        import traceback
        traceback.print_exc()
        # Return original conversation on error, but ensure it's properly encoded
        try:
            safe_conversation = []
            for msg in conversation:
                safe_msg = msg.copy()
                if 'content' in safe_msg:
                    content = safe_msg['content']
                    if isinstance(content, bytes):
                        safe_msg['content'] = content.decode('utf-8', errors='replace')
                    elif not isinstance(content, str):
                        safe_msg['content'] = str(content)
                safe_conversation.append(safe_msg)
            return safe_conversation
        except Exception:
            return conversation


def _is_likely_english(text: str) -> bool:
    """Simple heuristic to check if text is likely in English."""
    if not text:
        return True
    
    # Count English words vs Arabic characters
    english_words = ['the', 'and', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 
                    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
                    'patient', 'medical', 'doctor', 'health', 'symptoms', 'analysis']
    
    text_lower = text.lower()
    english_word_count = sum(1 for word in english_words if word in text_lower)
    
    # Check for Arabic characters (simplified)
    arabic_char_count = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
    
    # If we find common English words and no Arabic characters, likely English
    return english_word_count > 0 and arabic_char_count == 0
