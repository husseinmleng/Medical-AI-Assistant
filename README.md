# Medical AI Assistant

A comprehensive AI-powered medical assistant that provides breast cancer risk assessment, X-ray image analysis, and medical report interpretation with voice interaction capabilities.

## Features

### 🗣️ **Chat with LLM**
- Natural language conversation with an empathetic AI medical assistant
- Support for both English and Arabic (Egyptian dialect)
- Context-aware conversations that remember your medical history

### 📄 **Chat with Medical Reports**
- Upload and interpret medical documents (PDF, DOCX, JPG, PNG)
- Get patient-friendly explanations of medical findings
- Ask follow-up questions about your reports

### 🩻 **Chat with X-Ray Images**
- Upload X-ray images for AI-powered analysis
- Get detailed analysis results with confidence scores
- Download annotated images highlighting areas of interest

### 🎤 **Voice Functionality**
- Voice-to-text transcription for hands-free interaction
- Text-to-speech for accessibility
- Support for multiple audio formats

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Breast-Cancer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

4. Run the application:
```bash
streamlit run lg_st_app.py
```

## Usage

1. **Start a New Chat**: Select your preferred language (English or Arabic)
2. **Upload X-Ray Images**: Use the X-Ray uploader to analyze medical images
3. **Upload Medical Reports**: Use the Reports uploader to interpret medical documents
4. **Voice Interaction**: Use the microphone button for voice input
5. **Chat Continuously**: Ask follow-up questions about your analysis results

## Architecture

- **Frontend**: Streamlit web application
- **Backend**: LangGraph-based conversational AI agent
- **AI Models**: OpenAI GPT-4 for conversation and analysis
- **Image Analysis**: YOLO model for X-ray interpretation
- **State Management**: Persistent conversation state with checkpointing

## Core Components

- `lg_st_app.py` - Main Streamlit application
- `src/graph.py` - LangGraph agent implementation
- `src/app_logic.py` - Core application logic and state management
- `src/tools.py` - Medical analysis tools and functions
- `src/reports_agent.py` - Medical report interpretation agent
- `src/yolo_model.py` - X-ray image analysis model

## Important Notes

- This assistant provides general guidance and is not a substitute for professional medical advice
- Always consult with qualified healthcare professionals for diagnosis and treatment
- The AI analysis results are for informational purposes only

## License

[Add your license information here]
