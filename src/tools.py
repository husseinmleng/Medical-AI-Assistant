# src/tools.py

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.yolo_model import detect_cancer_in_image
from dotenv import load_dotenv
import asyncio

load_dotenv()

@tool
def analyze_xray_image(image_path: str,ml_result: str) -> str:
    """
    Analyzes a medical X-ray image for signs of breast cancer using a YOLO model based on the ml result.
    This is the SECOND and FINAL step. Call this tool when the user provides an image file.
    It returns the analysis result, confidence, and the path to the annotated image.
    """
    print(f"Analyzing X-ray image at path: {image_path} with ml_result: {ml_result}")
    if isinstance(ml_result, str):
        print(f"DEBUG: ML Result (assessment) = {ml_result}")
    try:
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

    # Coalesce None values to safe defaults
    safe_ml_result = ml_result or "Not available"
    safe_xray_result = xray_result or "Not available"
    try:
        safe_ml_conf = float(ml_confidence) if ml_confidence is not None else 0.0
    except Exception:
        safe_ml_conf = 0.0
    try:
        safe_xray_conf = float(xray_confidence) if xray_confidence is not None else 0.0
    except Exception:
        safe_xray_conf = 0.0

    questionnaire_summary = format_questionnaire(questionnaire_inputs or {})
    prompt_en = f"""You are an empathetic AI medical assistant speaking to a patient in English.
Your task is to provide a clear, calm, and detailed summary of their two-part breast cancer risk assessment.

**Patient's Questionnaire Summary:**
{questionnaire_summary}

**Analysis Results:**
1.  **Questionnaire-Based Assessment:**
    -   Result: **{safe_ml_result}**
    -   Confidence: **{safe_ml_conf:.1f}%**
2.  **X-ray Image Analysis:**
    -   Result: **{safe_xray_result}**
    -   Confidence: **{safe_xray_conf:.1f}%**

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
   - النتيجة: **{('إيجابية' if safe_ml_result == 'Positive' else ('سلبية' if safe_ml_result == 'Negative' else 'غير متاح'))}**
   - نسبة الثقة: **{safe_ml_conf:.1f}%**
٢. **تحليل صورة الأشعة:**
   - النتيجة: **{('إيجابية' if safe_xray_result == 'Positive' else ('سلبية' if safe_xray_result == 'Negative' else 'غير متاح'))}**
   - نسبة الثقة: **{safe_xray_conf:.1f}%**

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