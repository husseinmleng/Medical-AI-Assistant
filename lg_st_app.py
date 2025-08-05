# st_app.py (Corrected)
import streamlit as st
import uuid
import asyncio
import os
import tempfile
from src.app_logic import run_graph, clear_session_history, generate_chat_title, transcribe_audio, get_history
from streamlit_mic_recorder import mic_recorder

# --- Page Configuration ---
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🩺",
    layout="wide"
)

# --- Styling ---
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    /* Sidebar styling */
    .stButton button {
        width: 100%;
        text-align: left;
        margin-bottom: 0.25rem;
    }
    /* Ensure chat messages wrap text */
    .st-emotion-cache-1c7y2kd {
        white-space: pre-wrap;
        word-wrap: break-word;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None

# --- Main Processing Function (Refactored) ---
def handle_chat_submission(user_input, image_path=None):
    """
    Central function to process all user inputs (text, audio, image).
    It runs the graph and updates the session state correctly.
    """
    session_id = st.session_state.active_session_id
    active_conv = st.session_state.conversations[session_id]
    
    # Use a display message for image uploads
    display_message = "Here is the X-ray image for analysis." if image_path else user_input
    
    # Add user message to the conversation state
    active_conv["messages"].append({"role": "user", "content": display_message})

    # Show a spinner while the backend is working
    with st.spinner("AI assistant is thinking..."):
        # Generate a title for new chats on the first user message
        is_new_chat = len(active_conv["messages"]) < 2
        if is_new_chat and not image_path:
            try:
                active_conv['title'] = asyncio.run(generate_chat_title(user_input))
            except Exception as e:
                print(f"Error generating title: {e}")
                active_conv['title'] = "Medical Chat"

        # Run the graph with the actual user input
        response = run_graph(user_input, session_id, active_conv["lang"], image_path)
        
        # CRITICAL: Replace the entire message history with the final, complete
        # history from the graph's state. This ensures consistency.
        final_messages = response.get("messages", [])
        active_conv["messages"] = [{"role": msg.type, "content": msg.content} for msg in final_messages]
        
        # Update the path for the annotated image if it exists
        active_conv["annotated_image_path"] = response.get("annotated_image_path")

    # Clean up the temporary file for the uploaded image
    if image_path and "temp_upload" in image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)
            print(f"Removed temporary upload file: {image_path}")
        except OSError as e:
            print(f"Error removing temporary upload file {image_path}: {e}")

# --- Helper Functions ---
def create_new_chat():
    """Creates a new chat session."""
    session_id = str(uuid.uuid4())
    st.session_state.active_session_id = session_id
    st.session_state.conversations[session_id] = {
        "title": "New Chat",
        "lang": None,
        "messages": [],
        "annotated_image_path": None,
    }
    clear_session_history(session_id)
    return session_id

def switch_conversation(session_id):
    """Switches the active conversation and reloads its history."""
    st.session_state.active_session_id = session_id
    active_conv = st.session_state.conversations[session_id]
    history_messages = get_history(session_id)
    active_conv['messages'] = [{"role": msg.type, "content": msg.content} for msg in history_messages]

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
        if st.button(conv_data["title"], key=f"conv_{session_id}", use_container_width=True, type=button_type):
            switch_conversation(session_id)
            st.rerun()

# --- Main Chat Interface ---
st.title("Medical AI Assistant")

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
    # --- Display Chat Messages ---
    for msg in active_conv["messages"]:
        role = "assistant" if msg["role"] in ["assistant", "ai"] else "user"
        with st.chat_message(role):
            st.write(msg["content"])

    # After displaying messages, check if there's an annotated image to show
    if active_conv.get("annotated_image_path") and os.path.exists(active_conv["annotated_image_path"]):
        with st.chat_message("assistant"):
            st.image(active_conv["annotated_image_path"], caption="Annotated X-ray Image")

    # --- Input Area ---
    st.markdown("---")

    image_path_key = f'image_path_to_process_{active_session_id}'

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your X-ray image here when prompted",
            type=['png', 'jpg', 'jpeg'],
            key=f"uploader_{active_session_id}"
        )
    with col2:
        analyze_button = st.button(
            "🔬 Analyze X-ray",
            disabled=uploaded_file is None,
            use_container_width=True
        )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1], prefix="temp_upload_") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state[image_path_key] = tmp_file.name

    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        text_prompt = st.chat_input("Type your message...", key=f"chat_input_{active_session_id}")
    with col2:
        audio_info = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", key=f'recorder_{active_session_id}', use_container_width=True)

    # --- INPUT HANDLING LOGIC (Simplified) ---

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

    elif analyze_button:
        image_path_to_process = st.session_state.get(image_path_key)
        if image_path_to_process and os.path.exists(image_path_to_process):
            handle_chat_submission(
                user_input="Here is the X-ray image for analysis.",
                image_path=image_path_to_process
            )
            if image_path_key in st.session_state:
                del st.session_state[image_path_key]
            st.rerun()
        else:
            st.error("Could not find the uploaded image. Please upload it again.")
