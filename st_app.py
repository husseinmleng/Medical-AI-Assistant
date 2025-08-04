# st_app.py
import streamlit as st
import uuid
import asyncio
import os
import tempfile
from src.llm_chat import run_chat, clear_session, generate_chat_title, transcribe_audio
from src.yolo_model import detect_cancer_in_image
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
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None

# --- Helper Functions ---
def create_new_chat():
    """Creates a new chat session."""
    session_id = str(uuid.uuid4())
    st.session_state.active_session_id = session_id
    st.session_state.conversations[session_id] = {
        "title": "New Chat",
        "lang": None,
        "messages": []
    }
    # Initialize session-specific state keys
    st.session_state[f'last_audio_id_{session_id}'] = None
    st.session_state[f'last_file_id_{session_id}'] = None
    clear_session(session_id)
    return session_id

def switch_conversation(session_id):
    """Switches the active conversation."""
    st.session_state.active_session_id = session_id

if not st.session_state.conversations or st.session_state.active_session_id not in st.session_state.conversations:
    create_new_chat()
    st.rerun()

# --- Main Processing Function ---
def process_and_display_chat(user_input, session_id, lang, image_path=None):
    """Handles the chat logic and updates the UI."""
    active_conv = st.session_state.conversations[session_id]

    # Generate title for new chats
    is_new_chat = len(active_conv["messages"]) < 2

    with st.spinner("AI assistant is thinking..."):
        if is_new_chat:
            active_conv['title'] = asyncio.run(generate_chat_title(user_input))

        response = run_chat(user_input, session_id, lang=lang, image_path=image_path)

        if response['type'] == 'final_analysis':
            final_message = {
                "role": "assistant",
                "content": response['explanation'],
                "annotated_image_path": response['annotated_image_path']
            }
            active_conv["messages"].append(final_message)
        elif response['type'] == 'agent_message':
            active_conv["messages"].append({"role": "assistant", "content": response['content']})
        else: # Error
            st.error(response['content'])

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
        active_conv["messages"].append({"role": "assistant", "content": "Hello! I'm your AI Medical Assistant. how can I assist you today?"})
        st.rerun()
    if col2.button("🇸🇦 العربية", use_container_width=True, type="primary"):
        active_conv["lang"] = "ar"
        active_conv["messages"].append({"role": "assistant", "content": "مرحبًا! أنا مساعدك الطبي الذكي. كيف يمكنني مساعدتك اليوم؟"})
        st.rerun()
else:
    # --- Display Chat Messages ---
    for message in active_conv["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "user" and "image_path" in message and os.path.exists(message["image_path"]):
                st.image(message["image_path"], caption="Uploaded X-ray", use_column_width=True)
            elif message["role"] == "assistant" and "annotated_image_path" in message and message["annotated_image_path"] and os.path.exists(message["annotated_image_path"]):
                st.image(message["annotated_image_path"], caption="Annotated X-ray", use_column_width=True)

    # --- Input Area ---
    st.markdown("---")

    # Session-specific key for the uploaded image path
    image_path_key = f'image_path_to_process_{active_session_id}'

    # Layout for file uploader and analysis button
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your X-ray image here when prompted",
            type=['png', 'jpg', 'jpeg'],
            key=f"uploader_{active_session_id}"
        )

    with col2:
        # The button is disabled if no file is uploaded
        analyze_button = st.button(
            "🔬 Analyze X-ray",
            disabled=uploaded_file is None,
            use_container_width=True
        )

    # If a file is uploaded, save its path to the session state
    if uploaded_file is not None:
        try:
            # To avoid re-saving on every rerun, we check if it's already there
            if st.session_state.get(image_path_key) is None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    st.session_state[image_path_key] = tmp_file.name
        except Exception as e:
            st.error(f"Error saving uploaded file: {e}")

    # Layout for text and audio input
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        text_prompt = st.chat_input("Type your message...", key=f"chat_input_{active_session_id}")
    with col2:
        audio_info = mic_recorder(
            start_prompt="🎤",
            stop_prompt="⏹️",
            key=f'recorder_{active_session_id}',
            use_container_width=True
        )

    # --- INPUT HANDLING LOGIC ---

    # 1. Handle Text Input
    if text_prompt:
        active_conv["messages"].append({"role": "user", "content": text_prompt})
        process_and_display_chat(text_prompt, active_session_id, active_conv["lang"])
        st.rerun()

    # 2. Handle Audio Input
    elif audio_info and audio_info['id'] is not None:
        last_audio_id_key = f'last_audio_id_{active_session_id}'
        if audio_info['id'] != st.session_state.get(last_audio_id_key):
            st.session_state[last_audio_id_key] = audio_info['id']
            with st.spinner("🎙️ Transcribing audio..."):
                transcribed_text = asyncio.run(transcribe_audio(audio_info['bytes'], lang=active_conv["lang"]))
            if transcribed_text and transcribed_text.strip():
                active_conv["messages"].append({"role": "user", "content": transcribed_text})
                process_and_display_chat(transcribed_text, active_session_id, active_conv["lang"])
                st.rerun()
            else:
                st.toast("⚠️ Audio could not be transcribed. Please try speaking again.", icon="🎤")

    # 3. Handle "Analyze X-ray" Button Click
    elif analyze_button:
        image_path_to_process = st.session_state.get(image_path_key)
        if image_path_to_process and os.path.exists(image_path_to_process):
            user_input_for_image = "Here is the X-ray image for analysis."
            active_conv["messages"].append({"role": "user", "content": user_input_for_image, "image_path": image_path_to_process})
            process_and_display_chat(
                user_input_for_image,
                active_session_id,
                active_conv["lang"],
                image_path=image_path_to_process
            )
            # Clean up the stored path after processing
            del st.session_state[image_path_key]
            st.rerun()
        else:
            st.error("Could not find the uploaded image. Please upload it again.")
