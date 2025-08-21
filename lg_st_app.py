import streamlit as st
import uuid
import asyncio
import os
import tempfile
from datetime import datetime
from src.app_logic import (
    run_graph, clear_session_history, generate_chat_title, transcribe_audio, 
    get_history, reset_for_new_report, has_report_context, reset_for_new_xray
)
from src.html_agent import generate_html_report
from streamlit_mic_recorder import mic_recorder
from typing import List

# --- Page Configuration ---
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🩺",
    layout="wide"
)

# --- Styling ---
st.markdown(
    """
<style>
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    .stButton button {
        width: 100%;
        text-align: left;
        margin-bottom: 0.25rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }
    .ai-card {
        background: #f7fbff;
        border: 1px solid #e0eefc;
        padding: 12px 16px;
        border-radius: 12px;
    }
    .soft-divider { height: 4px; }
    /* Ensure chat messages wrap text */
    .stChatMessage p { white-space: pre-wrap; }
    .stChatMessage {
        border-radius: 12px;
        padding: 0.25rem 0.25rem;
    }
    .annotated-caption { color: #4c4c4c; font-size: 0.9rem; }
    .chip {
        display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.85rem;
        border: 1px solid #e5e7eb; background: #f9fafb; margin-right: 6px;
    }
    .chip.success { background:#ecfdf5; border-color:#a7f3d0; color:#065f46; }
    .chip.warn { background:#fffbeb; border-color:#fde68a; color:#92400e; }
    .chip.info { background:#eff6ff; border-color:#bfdbfe; color:#1e40af; }
    .muted { color:#6b7280; font-size:0.9rem; }
    .footer-note { color:#6b7280; font-size:0.8rem; text-align:center; margin-top: 12px; }
    .uploader-help { color:#6b7280; font-size:0.75rem; margin-top: -6px; }
    .report-context-indicator {
        background: #f0f9ff; border: 1px solid #bae6fd; 
        padding: 8px 12px; border-radius: 8px; margin-bottom: 12px;
        font-size: 0.9rem; color: #0c4a6e;
    }
    .sidebar-section {
        background: #f8fafc; border: 1px solid #e2e8f0;
        padding: 12px; border-radius: 8px; margin-bottom: 8px;
    }
    .sidebar-section h3 {
        margin-top: 0; margin-bottom: 8px; color: #374151;
        font-size: 0.9rem; font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Session State Initialization ---
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None

# --- Main Processing Function (Enhanced) ---
def handle_chat_submission(user_input, image_path=None, file_paths: List[str] = None):
    """
    Central function to process all user inputs (text, audio, image, files).
    Enhanced to handle report-based conversations properly and maintain state consistency.
    """
    session_id = st.session_state.active_session_id
    active_conv = st.session_state.conversations[session_id]
    
    print(f"\n🚀 CHAT SUBMISSION for session {session_id}:")
    print(f"  - User input: {user_input[:50] if user_input else 'None'}...")
    print(f"  - Image path: {image_path is not None}")
    print(f"  - File paths: {len(file_paths) if file_paths else 0} files")
    print(f"  - Current messages: {len(active_conv.get('messages', []))}")
    
    # Use a display message for image uploads
    if image_path:
        display_message = "Here is the X-ray image for analysis."
    elif file_paths:
        display_message = f"Here are {len(file_paths)} medical reports for interpretation."
    else:
        display_message = user_input
    
    # Add user message to the conversation state
    active_conv["messages"].append({"role": "user", "content": display_message})
    
    # Show a spinner while the backend is working
    with st.spinner("AI assistant is thinking..."):
        # Generate a title for new chats on the first user message
        is_new_chat = len(active_conv["messages"]) < 2
        if is_new_chat and not image_path and not file_paths:
            try:
                active_conv['title'] = asyncio.run(generate_chat_title(user_input))
            except Exception as e:
                print(f"Error generating title: {e}")
                active_conv['title'] = "Medical Chat"
        
        # Run the graph with the actual user input
        response = run_graph(user_input, session_id, active_conv["lang"], image_path, file_paths)
        
        if response.get("tts_audio"):
            st.session_state.audio_to_play = response["tts_audio"]
            
        # CRITICAL: Replace the entire message history with the final, complete
        # history from the graph's state. This ensures consistency.
        final_messages = response.get("messages", [])
        active_conv["messages"] = [{"role": msg.type, "content": msg.content} for msg in final_messages]
        
        # Update the path for the annotated image if it exists
        active_conv["annotated_image_path"] = response.get("annotated_image_path")
        
        # Store report context for indicating chat capability
        active_conv["has_report_context"] = bool(response.get("interpretation_result"))
        
        # Update the conversation title to reflect the type of analysis
        if image_path and not active_conv.get("title", "").startswith("X-Ray"):
            active_conv["title"] = f"X-Ray Analysis - {active_conv.get('title', 'Medical Chat')}"
        elif file_paths and not active_conv.get("title", "").startswith("Report"):
            active_conv["title"] = f"Report Analysis - {active_conv.get('title', 'Medical Chat')}"
    
    # Clean up the temporary file for the uploaded image
    if image_path and "temp_upload" in image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)
            print(f"Removed temporary upload file: {image_path}")
        except OSError as e:
            print(f"Error removing temporary upload file {image_path}: {e}")
    
    if file_paths:
        for fp in file_paths:
             if "temp_upload" in fp and os.path.exists(fp):
                try:
                    os.remove(fp)
                    print(f"Removed temporary upload file: {fp}")
                except OSError as e:
                    print(f"Error removing temporary upload file {fp}: {e}")

def sync_ui_state_with_backend(session_id: str):
    """Syncs the UI state with the backend state to ensure consistency."""
    try:
        from src.app_logic import get_session_state
        backend_state = get_session_state(session_id)
        
        if backend_state:
            # Update the conversation state to reflect backend state
            conv = st.session_state.conversations[session_id]
            
            # Update report context
            conv["has_report_context"] = bool(backend_state.get("interpretation_result"))
            
            # Update annotated image path
            if backend_state.get("annotated_image_path"):
                conv["annotated_image_path"] = backend_state.get("annotated_image_path")
            
            # Update conversation title based on analysis type
            if backend_state.get("interpretation_result") and not conv.get("title", "").startswith("Report"):
                conv["title"] = f"Report Analysis - {conv.get('title', 'Medical Chat')}"
            elif backend_state.get("xray_result") and not conv.get("title", "").startswith("X-Ray"):
                conv["title"] = f"X-Ray Analysis - {conv.get('title', 'Medical Chat')}"
                
            print(f"Synced UI state for session {session_id}")
    except Exception as e:
        print(f"Error syncing UI state: {e}")

# --- Helper Functions ---
def is_technical_message(content: str) -> bool:
    """Check if a message contains technical/raw output that shouldn't be displayed to users."""
    content_str = str(content)
    
    # If the message contains substantial user-friendly content, don't filter it out
    # even if it has technical elements
    user_friendly_indicators = [
        "النتيجة:", "النتيجة", "الثقة:", "الثقة", "الشرح:", "الشرح",
        "Result:", "Result", "Confidence:", "Confidence", "Explanation:", "Explanation",
        "بناءً على", "Based on", "مهم", "important", "يرجى", "Please"
    ]
    
    has_user_friendly_content = any(indicator in content_str for indicator in user_friendly_indicators)
    
    # Only filter out if it's purely technical
    if has_user_friendly_content:
        return False
    
    # Check for purely technical indicators
    technical_indicators = [
        "INITIAL_ASSESSMENT_RESULT:",
        "XRAY_RESULT:",
        "ANNOTATED_IMAGE_PATH:",
        "INTERPRETATION_RESULT:",
        "Error analyzing image:",
        "Error: There was a problem processing"
    ]
    
    # Only filter out if it contains technical indicators AND no user-friendly content
    return any(indicator in content_str for indicator in technical_indicators)

def create_new_chat():
    """Creates a new chat session."""
    session_id = str(uuid.uuid4())
    st.session_state.active_session_id = session_id
    st.session_state.conversations[session_id] = {
        "title": "New Chat",
        "lang": None,
        "messages": [],
        "annotated_image_path": None,
        "has_report_context": False,
    }
    # Clear the backend session history for the new chat
    clear_session_history(session_id)
    return session_id

def switch_conversation(session_id):
    """Switches the active conversation and reloads its history."""
    st.session_state.active_session_id = session_id
    active_conv = st.session_state.conversations[session_id]
    history_messages = get_history(session_id)
    active_conv['messages'] = [{"role": msg.type, "content": msg.content} for msg in history_messages]
    # Check if this session has report context
    active_conv["has_report_context"] = has_report_context(session_id)
    # Sync UI state with backend state
    sync_ui_state_with_backend(session_id)

def handle_analysis_type_switch(session_id: str, analysis_type: str, preserve_lang: str):
    """Handles switching between different analysis types while preserving conversation context."""
    # Get current conversation to preserve messages
    active_conv = st.session_state.conversations[session_id]
    current_messages = active_conv.get("messages", [])
    
    if analysis_type == "xray":
        # For X-ray analysis, only clear X-ray-specific results, preserve conversation and reports
        reset_for_new_xray(session_id, preserve_lang=preserve_lang)
        # Update UI state to reflect the change but PRESERVE messages
        active_conv["annotated_image_path"] = None
        active_conv["messages"] = current_messages  # Preserve conversation history
        print(f"Switched to X-ray analysis for session {session_id} (preserving {len(current_messages)} messages)")
    elif analysis_type == "report":
        # For report analysis, only clear report-specific results, preserve conversation and X-ray
        reset_for_new_report(session_id, preserve_lang=preserve_lang)
        # Update UI state to reflect the change but PRESERVE messages
        active_conv["has_report_context"] = False
        active_conv["messages"] = current_messages  # Preserve conversation history
        print(f"Switched to report analysis for session {session_id} (preserving {len(current_messages)} messages)")

def ensure_state_consistency():
    """Ensures consistency between frontend and backend state."""
    if st.session_state.active_session_id:
        try:
            # Sync the current session state
            sync_ui_state_with_backend(st.session_state.active_session_id)
        except Exception as e:
            print(f"Error ensuring state consistency: {e}")

def debug_state_info(session_id: str):
    """Debug function to display current state information."""
    if st.checkbox("🔍 Debug State Info", key=f"debug_{session_id}"):
        try:
            from src.app_logic import get_session_state
            backend_state = get_session_state(session_id)
            frontend_conv = st.session_state.conversations[session_id]
            
            st.write("**Backend State:**")
            st.json(backend_state)
            
            st.write("**Frontend Conversation State:**")
            st.json(frontend_conv)
            
        except Exception as e:
            st.error(f"Error getting debug info: {e}")

def generate_and_download_html_report(session_id: str, active_conv: dict):
    """Generates HTML report for the current conversation and analysis."""
    try:
        from src.app_logic import get_session_state
        backend_state = get_session_state(session_id) or {}
        
        # Prepare conversation data (filter out technical messages)
        conversation = []
        print(f"🔍 Processing {len(active_conv.get('messages', []))} messages for HTML report")
        for i, msg in enumerate(active_conv.get("messages", [])):
            is_technical = is_technical_message(msg["content"])
            print(f"🔍 Message {i}: role='{msg.get('role')}', technical={is_technical}, content_preview='{str(msg.get('content', ''))[:100]}...'")
            if not is_technical:
                conversation.append(msg)
                print(f"✅ Added message {i} to conversation")
            else:
                print(f"❌ Filtered out message {i} (technical)")
        
        print(f"🔍 Final conversation has {len(conversation)} messages")
        
        # Prepare patient info
        patient_info = {
            "session_id": session_id,
            "language": active_conv.get("lang", "en"),
            "date_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "conversation_length": len(conversation)
        }
        
        # Extract ML results from backend state or conversation if available
        ml_result = backend_state.get("ml_result", "N/A")
        ml_confidence = backend_state.get('ml_confidence', 0)
        
        # Extract report interpretation from backend state or conversation
        interpretation_result = backend_state.get("interpretation_result", "N/A")
        reports_context = backend_state.get("reports_text_context", "N/A")
        
        # If not in backend state, try to extract from conversation history
        if ml_result == "N/A" or ml_result is None:
            for msg in conversation:
                if msg["role"] == "assistant" and "INITIAL_ASSESSMENT_RESULT:" in msg["content"]:
                    # Extract from technical line
                    content = msg["content"]
                    if "INITIAL_ASSESSMENT_RESULT:" in content:
                        try:
                            tech_start = content.find("INITIAL_ASSESSMENT_RESULT:")
                            tech_segment = content[tech_start:].splitlines()[0]
                            parts = {p.split(':', 1)[0]: p.split(':', 1)[1] for p in tech_segment.split('|') if ':' in p}
                            ml_result = parts.get("INITIAL_ASSESSMENT_RESULT", "N/A")
                            conf_str = parts.get("CONFIDENCE")
                            if conf_str:
                                ml_confidence = float(conf_str)
                        except Exception:
                            pass
                    break
        
        # If interpretation not in backend, look for it in conversation
        if interpretation_result == "N/A" or interpretation_result is None:
            for msg in conversation:
                if msg["role"] == "assistant":
                    content = msg["content"]
                    # Look for report interpretation patterns
                    if ("report" in content.lower() and 
                        ("interpretation" in content.lower() or "analysis" in content.lower() or 
                         "findings" in content.lower() or "results" in content.lower()) and
                        len(content) > 100):  # Substantial content
                        # Skip technical messages
                        if not is_technical_message(content):
                            interpretation_result = content
                            break
        
        # Prepare analysis results from backend state
        analysis_results = {
            "ml_result": ml_result,
            "ml_confidence": f"{ml_confidence:.1f}%" if ml_confidence is not None else "N/A",
            "xray_result": backend_state.get("xray_result", "N/A"),
            "xray_confidence": f"{backend_state.get('xray_confidence', 0):.1f}%" if backend_state.get('xray_confidence') is not None else "N/A",
            "annotated_image_path": backend_state.get("annotated_image_path"),
            "interpretation_result": interpretation_result,
            "reports_context": reports_context
        }
        
        # Generate report title based on language
        lang = active_conv.get("lang", "en")
        if lang == "ar":
            report_title = "تقرير التحليل الطبي"
            patient_name = "المريض"
        else:
            report_title = "Medical Analysis Report"
            patient_name = "Patient"
        
        # Debug: Print analysis results for troubleshooting
        print(f"🔍 Analysis results for HTML report:")
        for key, value in analysis_results.items():
            if key == "interpretation_result" and value != "N/A":
                print(f"  {key}: {value[:100]}..." if len(str(value)) > 100 else f"  {key}: {value}")
            elif key == "reports_context" and value != "N/A":
                print(f"  {key}: {len(str(value))} characters" if value else "  {key}: No context")
            else:
                print(f"  {key}: {value}")
        
        # Debug: Check if we have any report-related content
        has_reports = any("report" in msg["content"].lower() for msg in conversation if msg["role"] == "assistant")
        print(f"📄 Report content detected in conversation: {has_reports}")
        
        # Generate HTML report
        html_content = generate_html_report(
            conversation=conversation,
            patient_info=patient_info,
            analysis_results=analysis_results,
            patient_name=patient_name,
            report_title=report_title,
            lang=lang
        )
        
        return html_content
        
    except Exception as e:
        st.error(f"Error generating HTML report: {str(e)}")
        return None

def display_analysis_status(active_conv):
    """Displays the current analysis status to help users understand what's available."""
    status_indicators = []
    
    # Get backend state for accurate status
    try:
        from src.app_logic import get_session_state
        backend_state = get_session_state(st.session_state.active_session_id)
        
        # Check for ML assessment results from backend
        if backend_state and backend_state.get("ml_result"):
            status_indicators.append(f"📊 ML Assessment: {backend_state.get('ml_result')}")
        
        # Check for X-ray results from backend
        if backend_state and backend_state.get("annotated_image_path"):
            status_indicators.append("🩻 X-Ray Analysis Available")
        
        # Check for report interpretation from backend
        if backend_state and (backend_state.get("interpretation_result") or backend_state.get("reports_text_context")):
            status_indicators.append("📄 Report Interpretation Available")
    except:
        # Fallback to frontend state if backend check fails
        if active_conv.get("ml_result"):
            status_indicators.append(f"📊 ML Assessment: {active_conv.get('ml_result')}")
        
        if active_conv.get("annotated_image_path"):
            status_indicators.append("🩻 X-Ray Analysis Available")
        
        if active_conv.get("has_report_context"):
            status_indicators.append("📄 Report Interpretation Available")
    
    if status_indicators:
        status_display = (
            '<div class="report-context-indicator">'
            '🔍 <strong>Available Analysis:</strong> ' + ' | '.join(status_indicators) +
            '</div>'
        )
        st.markdown(status_display, unsafe_allow_html=True)

# --- Ensure a chat is always active ---
if not st.session_state.active_session_id or st.session_state.active_session_id not in st.session_state.conversations:
    create_new_chat()
    st.rerun()

# --- Sidebar for Chat History ---
with st.sidebar:
    st.header("💬 Chat History")
    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()
    st.divider()
    
    sorted_conversations = sorted(st.session_state.conversations.items(), key=lambda item: item[0], reverse=True)
    
    for session_id, conv_data in sorted_conversations:
        button_type = "primary" if session_id == st.session_state.active_session_id else "secondary"
        
        # Add indicator for report-enabled chats
        title_display = conv_data["title"]
        if conv_data.get("has_report_context", False):
            title_display = f"📄 {title_display}"
            
        if st.button(title_display, key=f"conv_{session_id}", use_container_width=True, type=button_type):
            switch_conversation(session_id)
            st.rerun()
    
    # --- HTML Report Generation Section ---
    st.divider()
    st.markdown(
        '<div class="sidebar-section">'
        '<h3>📄 Medical Report</h3>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Check if there's any content to generate a report from
    active_conv = st.session_state.conversations[st.session_state.active_session_id]
    has_conversation = len(active_conv.get("messages", [])) > 1
    
    if has_conversation:
        if st.button("📄 Generate HTML Report", use_container_width=True, type="secondary"):
            with st.spinner("Generating HTML report..."):
                html_content = generate_and_download_html_report(st.session_state.active_session_id, active_conv)
                
                if html_content:
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    lang = active_conv.get("lang", "en")
                    filename = f"medical_report_{timestamp}.html"
                    
                    # Provide download button
                    st.download_button(
                        label="📥 Download Report",
                        data=html_content,
                        file_name=filename,
                        mime="text/html",
                        use_container_width=True
                    )
                    
                    st.success("✅ Report generated!")
                    
                    # Optional: Show preview link
                    with st.expander("📖 Preview", expanded=False):
                        st.components.v1.html(html_content, height=400, scrolling=True)
                        
        # Show status of what will be included in the report
        from src.app_logic import get_session_state
        backend_state = get_session_state(st.session_state.active_session_id) or {}
        report_items = []
        
        if backend_state.get("ml_result"):
            report_items.append("🧠 ML Assessment")
        if backend_state.get("xray_result"):
            report_items.append("🩻 X-Ray Analysis")
        if backend_state.get("interpretation_result"):
            report_items.append("📄 Report Interpretation")
        
        if report_items:
            st.caption(f"Will include: {', '.join(report_items)}")
        else:
            st.caption("📝 Conversation transcript will be included")
    else:
        st.button("📄 Generate HTML Report", use_container_width=True, disabled=True, type="secondary")
        st.caption("💬 Start a conversation first")

# --- Main Chat Interface ---
st.title("Medical AI Assistant")
st.caption("Friendly guidance, not a diagnosis. Always consult your doctor for medical decisions.")

# Ensure state consistency
ensure_state_consistency()

active_session_id = st.session_state.active_session_id
active_conv = st.session_state.conversations[active_session_id]

# --- Language Selection ---
if active_conv["lang"] is None:
    st.markdown("### Please select your preferred language:")
    col1, col2 = st.columns(2)
    if col1.button("🇺🇸 English", use_container_width=True, type="primary"):
        active_conv["lang"] = "en"
        active_conv["messages"].append({"role": "assistant", "content": "Hello! I'm your AI Medical Assistant. To start, I'll need to ask you a few questions about your health history. How can I assist you today?"})
        st.rerun()
    if col2.button("🇪🇬 العربي",  use_container_width=True, type="primary"):
        active_conv["lang"] = "ar"
        active_conv["messages"].append({"role": "assistant", "content": "أهلا بيكي انا مساعدك الطبي الذكي. هبدأ أسألك شوية أسئلة عن تاريخك الصحي. إزاي ممكن أساعدك النهاردة؟"})
        st.rerun()
else:
    # --- Show analysis status and report context indicators ---
    display_analysis_status(active_conv)
    
    # Show report context indicator if available
    if active_conv.get("has_report_context", False):
        st.markdown(
            '<div class="report-context-indicator">'
            '📄 <strong>Report Chat Mode:</strong> You can ask questions about your uploaded medical reports.'
            '</div>', 
            unsafe_allow_html=True
        )
    
    # Debug state information (only show in development)  
    # Commented out for production
    # debug_state_info(active_session_id)
    
    # --- Chat stream (single view) ---
    for msg in active_conv["messages"]:
        if is_technical_message(msg["content"]):
            continue
        role = "assistant" if msg["role"] in ["assistant", "ai"] else "user"
        avatar = "🩺" if role == "assistant" else "👤"
        with st.chat_message(role, avatar=avatar):
            st.write(msg["content"])
            
    # Show annotated image inline if exists
    if active_conv.get("annotated_image_path") and os.path.exists(active_conv["annotated_image_path"]):
        with st.chat_message("assistant", avatar="🩺"):
            st.image(active_conv["annotated_image_path"], caption="Annotated X-ray Image")
            try:
                with open(active_conv["annotated_image_path"], "rb") as f:
                    img_bytes = f.read()
                st.download_button(
                    label="Download annotated image",
                    data=img_bytes,
                    file_name=os.path.basename(active_conv["annotated_image_path"]),
                    mime="image/png",
                    use_container_width=True,
                )
            except Exception:
                pass

    if "audio_to_play" in st.session_state and st.session_state.audio_to_play:
        st.audio(st.session_state.audio_to_play, autoplay=True)
        st.session_state.audio_to_play = None

    st.markdown("---")
    # --- Inline attachments (same view as chat) ---
    image_path_key = f'image_path_to_process_{active_session_id}'
    report_paths_key = f'report_paths_to_process_{active_session_id}'
    up_left, up_right = st.columns([0.5, 0.5])
    with up_left:
        st.subheader("🩻 X-Ray")
        uploaded_file = st.file_uploader(
            "Upload your X-ray image",
            type=['png', 'jpg', 'jpeg'],
            key=f"uploader_{active_session_id}"
        )
        st.markdown("<div class='uploader-help'>PNG/JPG up to ~10MB</div>", unsafe_allow_html=True)
        analyze_xray_button = st.button(
            "🔬 Analyze X-ray",
            disabled=uploaded_file is None,
            use_container_width=True,
            key=f"analyze_btn_{active_session_id}"
        )
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1], prefix="temp_upload_") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state[image_path_key] = tmp_file.name
    with up_right:
        st.subheader("📄 Reports")
        uploaded_reports = st.file_uploader(
            "Upload medical reports (PDF, DOCX, JPG, etc.)",
            type=['pdf', 'png', 'jpg', 'jpeg', 'docx'],
            accept_multiple_files=True,
            key=f"report_uploader_{active_session_id}"
        )
        if uploaded_reports:
            st.markdown("<div class='muted'>Files selected:</div>", unsafe_allow_html=True)
            for r in uploaded_reports:
                st.markdown(f"<span class='chip info'>{r.name}</span>", unsafe_allow_html=True)
        
        # Show different button text based on context
        has_existing_reports = active_conv.get("has_report_context", False)
        button_text = "📄 Upload New Reports" if has_existing_reports else "📄 Interpret Reports"
        
        interpret_button = st.button(
            button_text,
            disabled=not uploaded_reports,
            use_container_width=True,
            key=f"interpret_btn_{active_session_id}"
        )
        if uploaded_reports:
            report_paths = []
            for report in uploaded_reports:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(report.name)[1], prefix="temp_upload_") as tmp_file:
                    tmp_file.write(report.getvalue())
                    report_paths.append(tmp_file.name)
            st.session_state[report_paths_key] = report_paths

    # --- Chat input with mic (same view) ---
    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        # Show different placeholder based on context
        placeholder = "Ask about your reports..." if active_conv.get("has_report_context", False) else "Type your message..."
        text_prompt = st.chat_input(placeholder, key=f"chat_input_{active_session_id}")
    with col2:
        audio_info = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", key=f'recorder_{active_session_id}', use_container_width=True)

    # --- INPUT HANDLING LOGIC (Enhanced) ---
    if text_prompt:
        handle_chat_submission(text_prompt)
        st.rerun()
    elif audio_info and audio_info['id'] is not None:
        last_audio_id_key = f'last_audio_id_{active_session_id}'
        if audio_info['id'] != st.session_state.get(last_audio_id_key):
            st.session_state[last_audio_id_key] = audio_info['id']
            with st.spinner("🎙️ Transcribing audio..."):
                try:
                    transcribed_text = asyncio.run(transcribe_audio(audio_info['bytes'], lang=active_conv["lang"]))
                    if transcribed_text and transcribed_text.strip():
                        handle_chat_submission(transcribed_text)
                        st.rerun()
                    else:
                        st.error("🎤 Audio transcription failed. Please try speaking again with a clear voice.")
                except Exception as e:
                    st.error(f"🎤 Audio transcription error: {str(e)}")
    
    elif analyze_xray_button:
        image_path_to_process = st.session_state.get(image_path_key)
        if image_path_to_process and os.path.exists(image_path_to_process):
            # Handle the switch to X-ray analysis (preserve conversation)
            handle_analysis_type_switch(active_session_id, "xray", active_conv["lang"])
            
            # Submit the X-ray for analysis while preserving conversation context
            handle_chat_submission(
                user_input="Here is the X-ray image for analysis.",
                image_path=image_path_to_process
            )
            
            # Clean up temporary files
            if image_path_key in st.session_state:
                del st.session_state[image_path_key]
            print(f"🖼️ X-ray analysis initiated for session {active_session_id}")
            st.rerun()
        else:
            st.error("🚨 Could not find the uploaded image. Please upload it again.")
            
    elif interpret_button:
        report_paths_to_process = st.session_state.get(report_paths_key)
        if report_paths_to_process:
            # Handle the switch to report analysis (preserve conversation)
            handle_analysis_type_switch(active_session_id, "report", active_conv["lang"])
            
            # Submit the reports for analysis while preserving conversation context
            handle_chat_submission(
                user_input="Please interpret these medical reports.",
                file_paths=report_paths_to_process
            )
            
            # Clean up temporary files
            if report_paths_key in st.session_state:
                del st.session_state[report_paths_key]
            print(f"🗂️ Report analysis initiated for session {active_session_id} with {len(report_paths_to_process)} files")
            st.rerun()
        else:
            st.error("🚨 Could not find the uploaded reports. Please upload them again.")

    st.markdown("""
    <div class="footer-note">This assistant provides general guidance and is not a substitute for professional medical advice.</div>
    """, unsafe_allow_html=True)