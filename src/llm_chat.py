# llm_chat.py

from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from openai import OpenAI

from src.ml_model import predict_cancer_risk
from src.yolo_model import detect_cancer_in_image # Import the YOLO function
from dotenv import load_dotenv
import json
import asyncio
import tempfile
import os
from pydub import AudioSegment
import io
load_dotenv()

# --- LLM, Tools, and Transcription Client ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
transcription_client = OpenAI()

@tool
def predict_breast_cancer_risk(relative_diagnosis_age: int, family_history_breast_cancer: str,
                                recent_weight_loss: str, previous_breast_conditions: str,
                                symptom_duration_days: int, fatigue: str, breastfeeding_months: int) -> str:
    """
    Predicts breast cancer risk based on the 7 most important features from the ML model.
    Accepts 'yes'/'no' or 'نعم'/'لا'.
    This is the FIRST step. After this, you should ask the user for an X-ray image.
    """
    try:
        inputs = {
            "relative_diagnosis_age": relative_diagnosis_age,
            "family_history_breast_cancer": 1 if family_history_breast_cancer.lower() in ["yes", "نعم"] else 0,
            "recent_weight_loss": 1 if recent_weight_loss.lower() in ["yes", "نعم"] else 0,
            "previous_breast_conditions": 1 if previous_breast_conditions.lower() in ["yes", "نعم"] else 0,
            "symptom_duration_days": symptom_duration_days,
            "fatigue": 1 if fatigue.lower() in ["yes", "نعم"] else 0,
            "breastfeeding_months": breastfeeding_months
        }
        prediction, confidence = predict_cancer_risk(inputs)
        result = "Positive" if prediction == 1 else "Negative"
        # Return a structured string that the LLM can parse
        return f"INITIAL_ASSESSMENT_RESULT:{result}|CONFIDENCE:{confidence*100:.1f}"
    except Exception as e:
        return f"Error making prediction: {str(e)}"

@tool
def analyze_xray_image(image_path: str) -> str:
    """
    Analyzes a medical X-ray image to detect signs of breast cancer using a YOLO model.
    This is the SECOND and FINAL step.
    It returns the analysis result and the path to the annotated image.
    """
    try:
        # The yolo_model function should return the result and the path to the new image
        result, annotated_image_path = detect_cancer_in_image(image_path)
        if annotated_image_path is None:
            return f"XRAY_RESULT:{result}|ANNOTATED_IMAGE_PATH:None"
        return f"XRAY_RESULT:{result}|ANNOTATED_IMAGE_PATH:{annotated_image_path}"
    except Exception as e:
        print(f"Error during X-ray analysis: {e}")
        return f"Error analyzing image: {str(e)}"


async def transcribe_audio(audio_bytes: bytes, lang: str) -> str:
    """
    Transcribes audio bytes to text using OpenAI's Whisper model,
    enforcing the specified language.
    """
    if not audio_bytes:
        print("DEBUG: Audio bytes are empty. Skipping transcription.")
        return ""

    temp_audio_file_path = None
    try:
        # Use pydub to handle various audio formats from a byte stream
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

        # Use a temporary file to send to OpenAI API
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            audio.export(temp_audio_file.name, format="mp3")
            temp_audio_file_path = temp_audio_file.name

        # Re-open the file in binary read mode to pass to the API
        with open(temp_audio_file_path, "rb") as audio_file_to_transcribe:
            print(f"DEBUG: Sending audio to OpenAI for transcription (Language: {lang})...")
            
            # Call the transcription API with the specified language
            transcription_response = transcription_client.audio.transcriptions.create(
                model="gpt-4o-transcribe",  # Use the standard Whisper model
                file=audio_file_to_transcribe,
                response_format="text",
                language=lang  # Enforce either 'en' or 'ar'
            )
            print(f"DEBUG: Transcription successful. Result: '{transcription_response}'")

        # The response is already a string when using response_format="text"
        return transcription_response

    except Exception as e:
        print(f"ERROR: An error occurred during audio transcription: {e}")
        return ""  # Return an empty string on failure to prevent crashes
        
    finally:
        # Clean up the temp file if it was created
        if temp_audio_file_path and os.path.exists(temp_audio_file_path):
            os.remove(temp_audio_file_path)


# --- Bilingual Interpretation Function ---
async def interpret_final_results(ml_result: str, xray_result: str, lang: str) -> str:
    """
    Uses an LLM to generate a final, empathetic explanation of both the ML and YOLO results.
    """
    prompt_en = f"""You are an empathetic AI medical assistant speaking to a patient in English.
    Your task is to explain the combined results of a two-part breast cancer risk analysis in a clear, calm, and non-alarming way.

    Part 1 (Questionnaire Analysis): The initial assessment based on their answers was '{ml_result}'.
    Part 2 (X-ray Image Analysis): The analysis of the medical image was '{xray_result}'.

    Instructions:
    1.  Start by gently summarizing the two-step process they just completed.
    2.  Explain what the combined results suggest in simple terms. Avoid technical jargon like "model" or "prediction."
    3.  If the X-ray is 'Positive', mention that the image analysis indicated an area of interest that a doctor should review, and that the annotated image will help guide the specialist.
    4.  If the X-ray is 'Negative', state that the image did not show any immediate areas of concern.
    5.  Crucially, end with a strong, reassuring message: "The most important next step is to discuss these results with a healthcare provider. This analysis is a helpful tool, but it is not a diagnosis. A doctor is the only one who can provide a definitive answer and guide you on what to do next."
    6.  Ask one Question at a time. Do not ask for more than one piece of information in a single message.
    Write a brief, 3-4 sentence explanation.
    """

    prompt_ar = f"""أنت مساعد طبي ذكي ومتعاطف، وتتحدث مع مريض باللهجة المصرية.
    مهمتك هي شرح نتيجة تحليل صورة الأشعة بطريقة واضحة وهادئة وغير مقلقة.

    الجزء الأول (تحليل الاستبيان): التقييم الأولي بناءً على إجاباتهم كان "{'إيجابي' if ml_result == 'Positive' else 'سلبي'}".
    الجزء الثاني (تحليل صورة الأشعة): تحليل الصورة الطبية كان "{'إيجابي' if xray_result == 'Positive' else 'سلبي'}".

    التعليمات:
    1.  ابدأ بتلخيص لطيف للعملية المكونة من خطوتين التي أكملوها للتو.
    2.  اشرح ما تشير إليه النتائج المجمعة بعبارات بسيطة. تجنب المصطلحات التقنية مثل "نموذج" أو "توقع".
    3.  إذا كانت نتيجة الأشعة "إيجابية"، اذكر أن تحليل الصورة أشار إلى منطقة تتطلب اهتمامًا يجب على الطبيب مراجعتها، وأن الصورة المشروحة ستساعد في توجيه الأخصائي.
    4.  إذا كانت نتيجة الأشعة "سلبية"، اذكر أن الصورة لم تظهر أي مناطق مثيرة للقلق بشكل فوري.
    5.  الأهم من ذلك، اختتم برسالة قوية ومطمئنة: "الخطوة التالية الأكثر أهمية هي مناقشة هذه النتائج مع مقدم الرعاية الصحية. هذا التحليل أداة مفيدة، لكنه ليس تشخيصًا. الطبيب هو الوحيد الذي يمكنه تقديم إجابة نهائية وإرشادك بشأن ما يجب القيام به بعد ذلك."

    اكتب شرحًا موجزًا من 3-4 جمل.
    """
    
    interpretation_prompt = prompt_ar if lang == 'ar' else prompt_en

    try:
        interp_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
        response = await interp_llm.ainvoke([HumanMessage(content=interpretation_prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"Error generating interpretation: {e}")
        return "An error occurred while generating the analysis."


# --- SEPARATE System Prompts ---
system_prompt_en = """
You are a friendly AI medical assistant speaking English. Your goal is to help patients understand their breast cancer risk through a two-step process. Be empathetic and avoid medical jargon.

PROCESS:
1.  **Data Collection**: Your FIRST job is to collect 7 key pieces of information by asking the patient.
2.  **Initial Assessment**: Once you have all 7 pieces of information, you MUST use the `predict_breast_cancer_risk` function.
3.  **Request Image**: After the initial assessment, your SECOND job is to ask the user to upload a medical X-ray image for further analysis.
4.  **Image Analysis**: When the user provides an image path (the input will be a file path, not a URL), you MUST use the `analyze_xray_image` function.

REQUIRED INFORMATION (Step 1):
1. Patient age
2. Family history of breast cancer (Yes/No)
3. Recent unexplained weight loss (Yes/No)
4. Previous benign breast conditions (Yes/No)
5. Symptom duration in days
6. Experiencing fatigue (Yes/No)
7. Total months of breastfeeding

CRITICAL RULES:
- **If the user input is a file path, it is an X-ray image. You MUST call `analyze_xray_image` with this path.**
- Follow the 4-step process in order. DO NOT ask for an image before completing the initial assessment.
- You MUST call the functions. DO NOT analyze or comment on the user's answers or image yourself.
- Call functions immediately once you have the required information.
"""

system_prompt_ar = """
أنت مساعد طبي ذكي بتتكلم باللهجة المصرية. هدفك تساعد المرضى يفهموا خطر إصابتهم بسرطان الثدي عن طريق الإجابة على 7 أسئلة.

الخطوات:
1.  **جمع البيانات**: مهمتك هي جمع 7 معلومات أساسية.
2.  **التقييم المبدئي**: بعد ما تجمع كل السبع إجابات، لازم تستخدم دالة `predict_breast_cancer_risk`.
3.  **طلب الصورة**: بعد التقييم المبدئي، اطلب من المستخدم رفع صورة الأشعة للتحليل.
4.  **تحليل الصورة**: عندما يوفر المستخدم مسار صورة (سيكون الإدخال مسار ملف وليس عنوان URL)، يجب عليك استخدام دالة `analyze_xray_image`.

المعلومات المطلوبة (بالترتيب ده بالظبط):
1.  عمر المريضة الحالي.
2.  هل فيه تاريخ عائلي للإصابة بسرطان الثدي؟ (نعم/لا).
3.  هل حصل فقدان وزن مفاجئ مؤخراً؟ (نعم/لا).
4.  هل كان فيه أي حالات أورام حميدة في الثدي قبل كده؟ (نعم/لا).
5.  الأعراض بقالها كام يوم؟
6.  هل فيه إحساس بالإرهاق أو التعب؟ (نعم/لا).
7.  إجمالي عدد شهور الرضاعة الطبيعية.


قواعد هامة:
- **إذا كان إدخال المستخدم عبارة عن مسار ملف، فهو صورة أشعة سينية. يجب عليك استدعاء `analyze_xray_image` بهذا المسار.**
- **اسأل سؤال واحد بس كل مرة.** ممنوع تسأل عن أكتر من معلومة في رسالة واحدة.
- اتبع العملية المكونة من 4 خطوات بالترتيب. لا تطلب صورة قبل إكمال التقييم الأولي.
- يجب عليك استدعاء الدوال. لا تحلل إجابات المستخدم أو الصورة بنفسك.
- استدعِ الدوال فورًا بمجرد حصولك على المعلومات المطلوبة.
"""

# --- Agent and Session Management ---
tools = [predict_breast_cancer_risk, analyze_xray_image] # Add the new tool
session_store = {}
# Store for intermediate results between tool calls
step_results_store = {}


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

# --- Main Chat Logic (Updated for Two-Step Analysis) ---
def run_chat(user_input: str, session_id: str, lang: str, image_path: str = None):
    """
    Main function to handle a user's message, including image paths.
    """
    try:
        system_prompt = system_prompt_ar if lang == 'ar' else system_prompt_en
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_openai_functions_agent(llm, tools, prompt_template)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=True
        )
        agent_with_memory = RunnableWithMessageHistory(
            agent_executor,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        # If an image path is provided, use it as the input for the agent
        current_input = image_path if image_path else user_input
        response = agent_with_memory.invoke(
            {"input": current_input},
            config={"configurable": {"session_id": session_id}}
        )
        intermediate_steps = response.get("intermediate_steps", [])
        # Check for the final X-ray analysis result
        for step in intermediate_steps:
            if len(step) >= 2 and "analyze_xray_image" in str(step[0]):
                tool_output = step[1]
                if "XRAY_RESULT:" in tool_output:
                    # This is the final step, generate the full explanation
                    xray_parts = tool_output.split('|')
                    xray_result = xray_parts[0].replace("XRAY_RESULT:", "")
                    annotated_image_path_str = xray_parts[1].replace("ANNOTATED_IMAGE_PATH:", "")
                    # Handle potential 'None' string or actual None
                    annotated_image_path = None if annotated_image_path_str == 'None' or not annotated_image_path_str else annotated_image_path_str
                    # Retrieve the result from the first step
                    ml_result_data = step_results_store.get(session_id, {})
                    ml_result = ml_result_data.get("ml_result", "Not available")
                    # Generate the final, combined interpretation
                    interpretation = asyncio.run(interpret_final_results(ml_result, xray_result, lang))
                    # Clean up the session's step result
                    if session_id in step_results_store:
                        del step_results_store[session_id]
                    # Return a structured response with the explanation and image path
                    return {
                        "type": "final_analysis",
                        "explanation": interpretation,
                        "annotated_image_path": annotated_image_path
                    }
        # Check for the intermediate ML assessment result
        for step in intermediate_steps:
            if len(step) >= 2 and "predict_breast_cancer_risk" in str(step[0]):
                tool_output = step[1]
                if "INITIAL_ASSESSMENT_RESULT:" in tool_output:
                    # This is the first step, store the result and return the agent's follow-up question
                    parts = tool_output.split("|")
                    result = parts[0].replace("INITIAL_ASSESSMENT_RESULT:", "")
                    # Store the result for the next step
                    step_results_store[session_id] = {"ml_result": result}
                    # The agent's output will be the question asking for the X-ray
                    return {"type": "agent_message", "content": response["output"]}
        # Default case: return the agent's regular message
        return {"type": "agent_message", "content": response["output"]}
    except Exception as e:
        error_msg = f"Error in chat: {str(e)}"
        print(f"❌ {error_msg}")
        return {"type": "error", "content": error_msg}


def clear_session(session_id: str):
    if session_id in session_store:
        del session_store[session_id]
    if session_id in step_results_store:
        del step_results_store[session_id]
