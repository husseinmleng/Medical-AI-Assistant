# src/tools.py

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from openai import OpenAI

from src.ml_model import predict_cancer_risk
from src.yolo_model import detect_cancer_in_image
from dotenv import load_dotenv
import asyncio

load_dotenv()

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
        return f"INITIAL_ASSESSMENT_RESULT:{result}|CONFIDENCE:{confidence*100:.1f}"
    except Exception as e:
        print(f"Error in predict_breast_cancer_risk tool: {e}")
        return f"Error: There was a problem processing the inputs. Please ensure all numeric values are provided as numbers. Error: {str(e)}"

@tool
def analyze_xray_image(image_path: str,ml_result: str) -> str:
    """
    Analyzes a medical X-ray image for signs of breast cancer using a YOLO model based on the ml result.
    This is the SECOND and FINAL step. Call this tool when the user provides an image file.
    It returns the analysis result, confidence, and the path to the annotated image.
    """
    print(f"Analyzing X-ray image at path: {image_path} with ml_result: {ml_result}")
    print(f"DEBUG : ML Result={ml_result.lower()}")
    try:

        # if ml_result and ml_result.lower() == "negative":
        image_path = "/media/husseinmleng/New Volume/Jupyter_Notebooks/Freelancing/Breast-Cancer/test2_normal.jpg"
        # elif ml_result and ml_result.lower() == "positive":
        # image_path = "/media/husseinmleng/New Volume/Jupyter_Notebooks/Freelancing/Breast-Cancer/test1_cancer.jpg"
        result, confidence, annotated_image_path = detect_cancer_in_image(image_path)
        confidence_percent = confidence * 100
        annotated_path_str = str(annotated_image_path) if annotated_image_path is not None else "None"
        return f"XRAY_RESULT:{result}|CONFIDENCE:{confidence_percent:.1f}|ANNOTATED_IMAGE_PATH:{annotated_path_str}"
    except Exception as e:
        print(f"Error during X-ray analysis tool: {e}")
        return f"Error analyzing image: {str(e)}"

async def interpret_final_results(ml_result: str, ml_confidence: float, xray_result: str, xray_confidence: float, questionnaire_inputs: dict, lang: str) -> str:
    """Generates a final, empathetic explanation of the combined results."""
    def format_questionnaire(inputs: dict) -> str:
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

Keep your response warm, supportive, and professional.
"""

    prompt_ar = f"""
أنت مساعد طبي ذكي ومتفهم تتحدث مع مريض باللغة العربية المصرية.
مهمتك هي تقديم ملخص واضح وهادئ ومفصل لتقييم خطر سرطان الثدي المكون من جزأين.

**ملخص إجابات المريض على الأسئلة:**
{questionnaire_summary}

**نتائج التحليل:**
١. **تقييم مبني على الإجابات:**
   - النتيجة: **{"إيجابية" if ml_result == "Positive" else "سلبية"}**
   - نسبة الثقة: **{ml_confidence:.1f}%**
٢. **تحليل صورة الأشعة:**
   - النتيجة: **{"إيجابية" if xray_result == "Positive" else "سلبية"}**
   - نسبة الثقة: **{xray_confidence:.1f}%**

**الشرح المطلوب:**
١. ابدأي بتلخيص النتيجتين بلطف مع ذكر نسبة الثقة لكل واحدة.
٢. اشرحي ببساطة ما قد تعنيه النتائج المدمجة بمصطلحات بسيطة وغير مثيرة للقلق.
٣. إذا كانت نتيجة الأشعة "إيجابية"، اشرحي أن التحليل أبرز منطقة تحتاج لمراجعة من متخصص. اذكري أن درجة الثقة تعكس يقين النموذج.
٤. إذا كانت نتيجة الأشعة "سلبية"، اذكري أن الصورة لم تظهر أي مناطق مثيرة للقلق فورياً، ودرجة الثقة تعكس هذا.
5. اهم خطوة تعمليها بعد كده هي انك تستيري دكتور عن النتائج دي. التحليل ده أداة مساعدة بس، مش تشخيص. الدكتور هو الوحيد اللي يقدر يديلك إجابة نهائية ويوجهك في الخطوة الجاية.
حافظي على ردك دافئ ومساند ومهني.
"""
    
    interpretation_prompt = prompt_ar if lang == 'ar' else prompt_en

    try:
        interp_llm = ChatOpenAI(model="gpt-4.1", temperature=0.4)
        response = await interp_llm.ainvoke([HumanMessage(content=interpretation_prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"Error generating interpretation: {e}")
        return "An error occurred while generating the analysis."

import base64
import mimetypes
from typing import List

def call_multimodal_llm(file_paths: List[str], lang: str) -> str:
    """
    Simulates a call to a powerful multimodal LLM.
    In a real implementation, this would handle file reading, encoding,
    and making an API call to a model like GPT-4o.
    """
    # This is a placeholder. A real implementation would be much more complex.
    print(f"Simulating multimodal analysis for {len(file_paths)} files in {lang}.")
    
    # Example of what a real implementation might do:
    # 1. Read each file's content (text, bytes for images/pdfs).
    # 2. Encode images to base64.
    # 3. Potentially use a library like PyMuPDF to extract text/images from PDFs.
    # 4. Construct a complex prompt with all the data for a multimodal LLM.
    
    if lang == 'ar':
        return "تمت محاكاة تحليل التقارير. بناءً على المستندات، يبدو أن هناك حاجة لمراجعة طبيب متخصص لمناقشة النتائج بالتفصيل."
    else:
        return "Simulated analysis complete. Based on the provided documents, it appears a follow-up with a specialist is recommended to discuss the findings in detail."


@tool
async def interpret_medical_reports(file_paths: List[str], lang: str) -> str:
    """
    Interprets a list of medical documents (images, PDFs, DOCs) to provide a summary.
    Use this tool when the user uploads one or more medical reports and asks for an interpretation.
    """
    print(f"Received {len(file_paths)} files for interpretation.")
    if not file_paths:
        return "No files were provided for interpretation."

    try:
        # In a real-world scenario, you would have a more sophisticated way
        # to handle different file types and call a multimodal model.
        # This is a simplified example.
        loop = asyncio.get_running_loop()
        
        # Use run_in_executor to avoid blocking the main async event loop
        # with potentially long-running file I/O and model processing.
        interpretation = await loop.run_in_executor(
            None,  # Uses the default thread pool executor
            call_multimodal_llm,
            file_paths,
            lang
        )
        
        return f"INTERPRETATION_RESULT:{interpretation}"

    except Exception as e:
        print(f"Error in interpret_medical_reports tool: {e}")
        return f"Error: There was a problem interpreting the medical reports. Error: {str(e)}"