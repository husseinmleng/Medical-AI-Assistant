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
    
    # Add user message to chat
    active_conv["messages"].append({"role": "user", "content": user_input})
    
    # Generate title for new chats
    is_new_chat = len(active_conv["messages"]) < 3 
    
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

    # Clean up temp file if one was created for an image
    if image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)
        except OSError as e:
            print(f"Error removing temporary file {image_path}: {e}")

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
    for msg in active_conv["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "annotated_image_path" in msg and msg["annotated_image_path"] and os.path.exists(msg["annotated_image_path"]):
                st.image(msg["annotated_image_path"], caption="Annotated X-ray Image")

    # --- Input Area ---
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Upload your X-ray image here when prompted", 
        type=['png', 'jpg', 'jpeg'],
        key=f"uploader_{active_session_id}"
    )
    
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
        process_and_display_chat(text_prompt, active_session_id, active_conv["lang"])
        st.rerun()

    # 2. Handle Audio Input
    elif audio_info and audio_info['id'] is not None:
        last_audio_id_key = f'last_audio_id_{active_session_id}'
        
        # Check if this is a new recording
        if audio_info['id'] != st.session_state.get(last_audio_id_key):
            # Store the new ID to prevent reprocessing
            st.session_state[last_audio_id_key] = audio_info['id']

            with st.spinner("🎙️ Transcribing audio..."):
                # Pass the selected language to the transcription function
                transcribed_text = asyncio.run(transcribe_audio(audio_info['bytes'], lang=active_conv["lang"]))
            
            if transcribed_text and transcribed_text.strip():
                process_and_display_chat(transcribed_text, active_session_id, active_conv["lang"])
                st.rerun()
            else:
                st.toast("⚠️ Audio could not be transcribed. Please try speaking again.", icon="🎤")

    # 3. Handle Image Upload
    elif uploaded_file is not None:
        last_file_id_key = f'last_file_id_{active_session_id}'
        
        # Create a unique identifier from the file's name and size.
        current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"

        # Check if this is a new file upload
        if current_file_id != st.session_state.get(last_file_id_key):
            # Store the new file ID to prevent reprocessing
            st.session_state[last_file_id_key] = current_file_id

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    # Store the path in session state
                    st.session_state.image_path_to_process = tmp_file.name
                
                # Set flag and rerun
                st.session_state.processing_image = True
                st.rerun()

            except Exception as e:
                st.error(f"Error processing image: {e}")

# --- Image Processing Logic (runs after rerun) ---
if st.session_state.get('processing_image'):
    # Retrieve the image path from session state
    image_path_to_process = st.session_state.get('image_path_to_process')
    active_session_id = st.session_state.active_session_id
    active_conv = st.session_state.conversations[active_session_id]

    if image_path_to_process and os.path.exists(image_path_to_process):
        # The user_input can be a generic message as the agent will prioritize the image path
        user_input_for_image = "Here is the X-ray image for analysis."
        
        # Call the main chat processing function, which will handle the agent logic
        process_and_display_chat(
            user_input_for_image, 
            active_session_id, 
            active_conv["lang"], 
            image_path=image_path_to_process
        )
    else:
        st.error("Something went wrong during image processing. Please try uploading again.")

    # Reset flags and clean up session state
    st.session_state.processing_image = False
    if 'image_path_to_process' in st.session_state:
        del st.session_state['image_path_to_process']
    
    st.rerun()
