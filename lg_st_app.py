import streamlit as st
import uuid
import asyncio
import os
import tempfile
from src.app_logic import (
    run_graph, clear_session_history, generate_chat_title, transcribe_audio, 
    get_history, reset_for_new_report, has_report_context, reset_for_new_xray,
    generate_and_download_report, generate_and_download_html_report # <-- IMPORT THE NEW FUNCTION
)
from streamlit_mic_recorder import mic_recorder
from typing import List

# --- LaTeX Installation Check ---
def check_latex_for_app():
    """Check if LaTeX is available for PDF generation."""
    try:
        from src.tools import check_latex_installation
        status = check_latex_installation()
        return status
    except Exception:
        return {"installed": False, "error": "Could not check LaTeX installation"}

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
    Enhanced to handle report-based conversations properly.
    """
    session_id = st.session_state.active_session_id
    active_conv = st.session_state.conversations[session_id]
    
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

# --- Helper Functions ---
def is_technical_message(content: str) -> bool:
    """Check if a message contains technical/raw output that shouldn't be displayed to users."""
    technical_indicators = [
        "INITIAL_ASSESSMENT_RESULT:",
        "XRAY_RESULT:",
        "CONFIDENCE:",
        "ANNOTATED_IMAGE_PATH:",
        "INTERPRETATION_RESULT:",
        "|",
        "Error analyzing image:",
        "Error: There was a problem processing"
    ]
    return any(indicator in content for indicator in technical_indicators)

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

    # --- NEW: PDF Report Download Section ---
    st.divider()
    st.header("📥 Download Report")
    
    # Check LaTeX installation and show warning if needed
    latex_status = check_latex_for_app()
    if not latex_status.get("installed", False):
        st.warning(
            f"⚠️ PDF generation requires LaTeX to be installed. "
            f"{latex_status.get('installation_guide', 'Please install a LaTeX distribution.')}"
        )
        if st.button("Check LaTeX Installation", key="check_latex_btn"):
            st.info(f"LaTeX Status: {latex_status}")
    
    # This section now handles the PDF generation and download
    if st.button("Generate PDF Report", key="generate_pdf_btn", use_container_width=True, disabled=not latex_status.get("installed", False)):
        active_id = st.session_state.active_session_id
        if active_id:
            async def _generate_pdf_report():
                with st.spinner("Generating professional PDF report..."):
                    try:
                        pdf_path = await generate_and_download_report(active_id)
                        if pdf_path and os.path.exists(pdf_path):
                            with open(pdf_path, "rb") as pdf_file:
                                pdf_bytes = pdf_file.read()
                            
                            # Use a session state key to store the bytes for the download button
                            st.session_state[f'pdf_bytes_{active_id}'] = pdf_bytes
                            st.session_state[f'pdf_filename_{active_id}'] = os.path.basename(pdf_path)
                            
                            # Clean up the server-side file after reading it
                            os.remove(pdf_path)
                            
                        else:
                            st.error("Failed to generate the PDF. Please check the logs.")
                            st.session_state[f'pdf_bytes_{active_id}'] = None

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.session_state[f'pdf_bytes_{active_id}'] = None
            asyncio.run(_generate_pdf_report())

    # Display the download button if the PDF bytes are available in session state
    active_id = st.session_state.active_session_id
    if st.session_state.get(f'pdf_bytes_{active_id}'):
        st.download_button(
            label="⬇️ Download PDF",
            data=st.session_state[f'pdf_bytes_{active_id}'],
            file_name=st.session_state[f'pdf_filename_{active_id}'],
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )

    if st.button("Generate HTML Report", key="generate_html_btn", use_container_width=True):
        active_id = st.session_state.active_session_id
        if active_id:
            async def _generate_html_report():
                with st.spinner("Generating HTML report..."):
                    try:
                        html_path = await generate_and_download_html_report(active_id)
                        if html_path and os.path.exists(html_path):
                            with open(html_path, "r", encoding="utf-8") as html_file:
                                html_bytes = html_file.read()
                            
                            st.session_state[f'html_bytes_{active_id}'] = html_bytes
                            st.session_state[f'html_filename_{active_id}'] = os.path.basename(html_path)
                            
                            os.remove(html_path)
                        else:
                            st.error("Failed to generate the HTML report. Please check the logs.")
                            st.session_state[f'html_bytes_{active_id}'] = None
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.session_state[f'html_bytes_{active_id}'] = None
            asyncio.run(_generate_html_report())

    if st.session_state.get(f'html_bytes_{active_id}'):
        st.download_button(
            label="⬇️ Download HTML",
            data=st.session_state[f'html_bytes_{active_id}'],
            file_name=st.session_state[f'html_filename_{active_id}'],
            mime="text/html",
            use_container_width=True,
            type="primary"
        )


# --- Main Chat Interface ---
st.title("Medical AI Assistant")
st.caption("Friendly guidance, not a diagnosis. Always consult your doctor for medical decisions.")
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
    if col2.button("🇸🇦 العربية", use_container_width=True, type="primary"):
        active_conv["lang"] = "ar"
        active_conv["messages"].append({"role": "assistant", "content": "مرحبًا! أنا مساعدك الطبي الذكي. لبدء التقييم، سأحتاج إلى طرح بعض الأسئلة حول تاريخك الصحي. كيف يمكنني مساعدتك اليوم؟"})
        st.rerun()
else:
    # --- Show report context indicator ---
    if active_conv.get("has_report_context", False):
        st.markdown(
            '<div class="report-context-indicator">'
            '📄 <strong>Report Chat Mode:</strong> You can ask questions about your uploaded medical reports.'
            '</div>', 
            unsafe_allow_html=True
        )
    
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
                transcribed_text = asyncio.run(transcribe_audio(audio_info['bytes'], lang=active_conv["lang"]))
            if transcribed_text and transcribed_text.strip():
                handle_chat_submission(transcribed_text)
                st.rerun()
            else:
                st.toast("⚠️ Audio could not be transcribed. Please try speaking again.", icon="🎤")
    
    elif analyze_xray_button:
        image_path_to_process = st.session_state.get(image_path_key)
        if image_path_to_process and os.path.exists(image_path_to_process):
            reset_for_new_xray(active_session_id, preserve_lang=active_conv["lang"]) 
            # Clear local chat view and any annotated image
            active_conv["messages"] = []
            active_conv["has_report_context"] = False
            handle_chat_submission(
                user_input="Here is the X-ray image for analysis.",
                image_path=image_path_to_process
            )
            if image_path_key in st.session_state:
                del st.session_state[image_path_key]
            st.rerun()
        else:
            st.error("Could not find the uploaded image. Please upload it again.")
            
    elif interpret_button:
        report_paths_to_process = st.session_state.get(report_paths_key)
        if report_paths_to_process:
            # Reset graph state for a new report while preserving current language
            reset_for_new_report(active_session_id, preserve_lang=active_conv["lang"]) 
            # Clear local chat view and any annotated image
            active_conv["messages"] = []
            active_conv["annotated_image_path"] = None
            active_conv["has_report_context"] = False
            handle_chat_submission(
                user_input="Please interpret these medical reports.",
                file_paths=report_paths_to_process
            )
            if report_paths_key in st.session_state:
                del st.session_state[report_paths_key]
            st.rerun()
        else:
            st.error("Could not find the uploaded reports. Please upload them again.")

    st.markdown("""
    <div class="footer-note">This assistant provides general guidance and is not a substitute for professional medical advice.</div>
    """, unsafe_allow_html=True)