
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from openai import OpenAI

from src.ml_model import predict_cancer_risk
from src.yolo_model import detect_cancer_in_image
from dotenv import load_dotenv
import asyncio
import tempfile
import os
from pydub import AudioSegment
import io

load_dotenv()

# --- LLM, Tools, and Transcription Client ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
transcription_client = OpenAI()

# --- Session & State Management ---
session_store = {}
# This dictionary will now be managed exclusively by the run_chat function
step_results_store = {}

@tool
def predict_breast_cancer_risk(relative_diagnosis_age: int, family_history_breast_cancer: str,
                               recent_weight_loss: str, previous_breast_conditions: str,
                               symptom_duration_days: int, fatigue: str, breastfeeding_months: int) -> str:
    """
    Predicts breast cancer risk based on 7 key patient factors.
    This is the FIRST step. After this, you MUST ask the user for an X-ray image.
    Use this tool once you have collected all 7 pieces of information.
    The user must provide a specific number for age, symptom duration, and breastfeeding months.
    For yes/no questions, the user must answer with 'yes' or 'no'.
    """
    try:
        # Convert yes/no strings to 1/0, being flexible with case and language
        inputs = {
            "relative_diagnosis_age": int(relative_diagnosis_age),
            "family_history_breast_cancer": 1 if family_history_breast_cancer.lower() in ["yes", "نعم"] else 0,
            "recent_weight_loss": 1 if recent_weight_loss.lower() in ["yes", "نعم"] else 0,
            "previous_breast_conditions": 1 if previous_breast_conditions.lower() in ["yes", "نعم"] else 0,
            "symptom_duration_days": int(symptom_duration_days),
            "fatigue": 1 if fatigue.lower() in ["yes", "نعم"] else 0,
            "breastfeeding_months": int(breastfeeding_months)
        }
        prediction, confidence = predict_cancer_risk(inputs)
        result = "Positive" if prediction == 1 else "Negative"
        # Return a structured string for easy parsing
        return f"INITIAL_ASSESSMENT_RESULT:{result}|CONFIDENCE:{confidence*100:.1f}"
    except Exception as e:
        print(f"Error in predict_breast_cancer_risk tool: {e}")
        return f"Error: There was a problem processing the inputs. Please ensure all numeric values are provided as numbers. Error: {str(e)}"

@tool
def analyze_xray_image(image_path: str) -> str:
    """
    Analyzes a medical X-ray image for signs of breast cancer using a YOLO model.
    This is the SECOND and FINAL step. Call this tool when the user provides an image file.
    It returns the analysis result, confidence, and the path to the annotated image.
    """
    try:
        result, confidence, annotated_image_path = detect_cancer_in_image(image_path)
        confidence_percent = confidence * 100
        
        # Ensure annotated_image_path is a string, even if None
        annotated_path_str = str(annotated_image_path) if annotated_image_path is not None else "None"
        
        return f"XRAY_RESULT:{result}|CONFIDENCE:{confidence_percent:.1f}|ANNOTATED_IMAGE_PATH:{annotated_path_str}"
    except Exception as e:
        print(f"Error during X-ray analysis tool: {e}")
        return f"Error analyzing image: {str(e)}"


async def transcribe_audio(audio_bytes: bytes, lang: str) -> str:
    """Transcribes audio to text using OpenAI's Whisper model."""
    if not audio_bytes:
        return ""
    temp_audio_file_path = None
    try:
        # Use an in-memory buffer
        audio_io = io.BytesIO(audio_bytes)
        audio_io.name = "temp_audio.mp3" # Whisper API needs a file name
        
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

async def interpret_final_results(ml_result: str, ml_confidence: float, xray_result: str, xray_confidence: float, questionnaire_inputs: dict, lang: str) -> str:
    """Generates a final, empathetic explanation of the combined results."""
    def format_questionnaire(inputs: dict) -> str:
        # A more readable format for the summary
        return "\n".join([f"- {key.replace('_', ' ').title()}: {value}" for key, value in inputs.items()])

    questionnaire_summary = format_questionnaire(questionnaire_inputs)

    prompt_en = f"""You are an empathetic AI medical assistant speaking to a patient in English.
Your task is to provide a clear, calm, and detailed summary of their two-part breast cancer risk assessment.

**Patient's Questionnaire Summary:**
{questionnaire_summary}

**Analysis Results:**
1.  **Questionnaire-Based Assessment:**
    -   Result: **{ml_result}**
    -   Confidence: **{ml_confidence:.1f}%**
2.  **X-ray Image Analysis:**
    -   Result: **{xray_result}**
    -   Confidence: **{xray_confidence:.1f}%**

**Your Explanation:**
1.  Start by gently summarizing the two results, mentioning the confidence level for each.
2.  Briefly explain what the combined results might suggest in simple, non-alarming terms.
3.  If the X-ray result is 'Positive', explain that the analysis highlighted an area of interest for a specialist to review. Mention that the confidence score reflects the model's certainty.
4.  If the X-ray result is 'Negative', state that the image did not show any immediate areas of concern, and the confidence score reflects this.
5.  **Crucially, end with this strong, reassuring message:** "The most important next step is to discuss these results with your healthcare provider. This analysis is a helpful tool, but it is not a diagnosis. A doctor is the only one who can provide a definitive answer and guide you on what to do next."
"""

    prompt_ar = f"""أهلاً بيكي مرة تانية. أنا هنا عشان أشرحلك نتايج التقييم بتاعك بالتفصيل. متقلقيش، هنمشي في الموضوع خطوة بخطوة.

**ملخص إجاباتك على الأسئلة:**
{questionnaire_summary}

**نتايج التحليل:**
١. **تقييم مبني على الإجابات:**
   - النتيجة: **{"إيجابية" if ml_result == "Positive" else "سلبية"}**
   - نسبة الثقة: **{ml_confidence:.1f}%**
٢. **تحليل صورة الأشعة:**
   - النتيجة: **{"إيجابية" if xray_result == "Positive" else "سلبية"}**
   - نسبة الثقة: **{xray_confidence:.1f}%**

**شرح النتايج:**
١. ابدأي بتلخيص النتيجتين بهدوء، مع ذكر نسبة الثقة لكل واحدة.
٢. اشرحي ببساطة إيه ممكن تكون دلالة النتايج دي مع بعض، من غير مصطلحات طبية معقدة.
٣. لو نتيجة الأشعة "إيجابية"، وضحي إن التحليل أظهر منطقة محتاجة اهتمام ومراجعة من دكتور متخصص. اذكري إن نسبة الثقة بتوضح مدى تأكد النموذج من النتيجة دي.
٤. لو نتيجة الأشعة "سلبية"، قولي إن الصورة موضحتش أي مناطق تدعو للقلق حاليًا، ونسبة الثقة بتعكس ده.
٥. **الأهم من كل ده، اختمي بالرسالة دي:** "أهم خطوة جاية هي إنك تتكلمي مع دكتورك وتناقشي معاه النتايج دي بالتفصيل. التحليل ده مجرد أداة مساعدة، لكنه مش تشخيص نهائي. الدكتور هو الوحيد اللي يقدر يديكي إجابة قاطعة ويوجهك للخطوات الجاية."
"""
    
    interpretation_prompt = prompt_ar if lang == 'ar' else prompt_en

    try:
        interp_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
        response = await interp_llm.ainvoke([HumanMessage(content=interpretation_prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"Error generating interpretation: {e}")
        return "An error occurred while generating the analysis."


# --- System Prompts ---
# FIX: Made the prompt more direct and checklist-oriented for the agent.
system_prompt_template = """
You are a calm, empathetic, and reassuring doctor speaking {language}. Your primary role is to guide a patient through a two-step breast cancer risk assessment.

**Your Persona:**
- **Warm & Welcoming:** Start with a kind greeting.
- **Conversational & Focused:** Ask ONE question at a time to gather the necessary information. Be natural.
- **Empathetic & Clear:** Acknowledge the user's answers and explain things simply.

**Your Process & Tool Use:**
1.  **Step 1: Gather Data for `predict_breast_cancer_risk`:**
    Your first job is to collect the following 7 pieces of information. Ask one question at a time until you have all of them:
    - `relative_diagnosis_age` (What was the age of the relative diagnosed with breast cancer?)
    - `family_history_breast_cancer` (Do you have a family history of breast cancer? yes/no)
    - `recent_weight_loss` (Have you experienced recent, unexplained weight loss? yes/no)
    - `previous_breast_conditions` (Have you had previous benign breast conditions? yes/no)
    - `symptom_duration_days` (For how many days have you been experiencing symptoms?)
    - `fatigue` (Are you experiencing unusual fatigue? yes/no)
    - `breastfeeding_months` (How many months in total have you breastfed?)
    Once you have answers for all 7, you **MUST** call the `predict_breast_cancer_risk` tool immediately.

2.  **Step 2: Request & Analyze X-ray:**
    After the first tool call, you **MUST** ask the user to upload their X-ray image. When they provide an image path (the input will contain 'Here is the X-ray image'), you **MUST** call the `analyze_xray_image` tool immediately.

**CRITICAL RULES:**
- You **MUST** call the tools when their required information is available. This is not optional.
- Ask only one question at a time.
- Do not make up results. Rely *only* on the tool outputs.
"""

def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

async def generate_chat_title(user_input: str) -> str:
    try:
        title_prompt = f"Create a very short, concise title (4 words max) for a medical chat conversation that starts with this user message: '{user_input}'. Do not use quotes in the title."
        title_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        response = await title_llm.ainvoke([HumanMessage(content=title_prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"Error generating title: {e}")
        return "New Chat"

def run_chat(user_input: str, session_id: str, lang: str, image_path: str = None):
    """Main function to handle a user's message, including image paths."""
    try:
        language_map = {"en": "English", "ar": "in an Egyptian dialect"}
        formatted_prompt = system_prompt_template.format(language=language_map.get(lang, "English"))
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", formatted_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        tools = [predict_breast_cancer_risk, analyze_xray_image]
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        
        agent_with_memory = RunnableWithMessageHistory(
            agent_executor,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        # Initialize session state if it doesn't exist
        if session_id not in step_results_store:
            step_results_store[session_id] = {}

        # If an image path is provided, it's the primary input for this turn.
        current_input = image_path if image_path else user_input
        
        response = agent_with_memory.invoke(
            {"input": current_input},
            config={"configurable": {"session_id": session_id}}
        )

        intermediate_steps = response.get("intermediate_steps", [])
        
        if not intermediate_steps:
            return {"type": "agent_message", "content": response.get("output", "")}

        # Process the latest tool call
        last_step = intermediate_steps[-1]
        tool_name = last_step[0].tool
        tool_output = last_step[1]

        if tool_name == 'predict_breast_cancer_risk':
            # Store the inputs and results from the first tool
            step_results_store[session_id]['questionnaire_inputs'] = last_step[0].tool_input
            # Safely parse the tool output
            if '|' in tool_output and ':' in tool_output:
                parts = {p.split(':', 1)[0]: p.split(':', 1)[1] for p in tool_output.split('|')}
                step_results_store[session_id]['ml_result'] = parts.get("INITIAL_ASSESSMENT_RESULT", "Error")
                step_results_store[session_id]['ml_confidence'] = float(parts.get("CONFIDENCE", 0.0))
            else:
                # Handle error string from tool
                step_results_store[session_id]['ml_result'] = "Error"
                step_results_store[session_id]['ml_confidence'] = 0.0

            # Return the agent's follow-up message (asking for the X-ray)
            return {"type": "agent_message", "content": response.get("output", "")}

        elif tool_name == 'analyze_xray_image':
            # This is the final step, generate the full explanation
            xray_parts = {p.split(':', 1)[0]: p.split(':', 1)[1] for p in tool_output.split('|')}
            xray_result = xray_parts.get("XRAY_RESULT", "Error")
            xray_confidence = float(xray_parts.get("CONFIDENCE", 0.0))
            annotated_image_path = xray_parts.get("ANNOTATED_IMAGE_PATH")
            if annotated_image_path == 'None':
                annotated_image_path = None

            # Retrieve all data from the session store
            session_data = step_results_store.get(session_id, {})
            ml_result = session_data.get("ml_result", "Not available")
            ml_confidence = session_data.get("ml_confidence", 0.0)
            questionnaire_inputs = session_data.get("questionnaire_inputs", {})

            interpretation = asyncio.run(interpret_final_results(
                ml_result, ml_confidence, xray_result, xray_confidence, questionnaire_inputs, lang
            ))
            
            clear_session(session_id)  # Clean up after final result
            
            return {
                "type": "final_analysis",
                "explanation": interpretation,
                "annotated_image_path": annotated_image_path
            }

        return {"type": "agent_message", "content": response.get("output", "")}

    except Exception as e:
        error_msg = f"An unexpected error occurred in the chat logic: {str(e)}"
        print(f"❌ {error_msg}")
        # Be careful with traceback in user-facing errors
        import traceback
        traceback.print_exc()
        return {"type": "error", "content": "I'm sorry, a system error occurred. Please try starting a new chat."}

def clear_session(session_id: str):
    """Clears the session data for a given session ID."""
    if session_id in session_store:
        del session_store[session_id]
    if session_id in step_results_store:
        del step_results_store[session_id]
    print(f"Cleared session data for {session_id}")
