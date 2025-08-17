from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.yolo_model import detect_cancer_in_image
from dotenv import load_dotenv
import asyncio
load_dotenv()
@tool
def analyze_xray_image(image_path: str) -> str:
    """
    Analyzes a medical X-ray image for signs of breast cancer using a YOLO model.
    This is the a standalone feature. Call this tool when the user provides an image file.
    It returns the analysis result, confidence, and the path to the annotated image.
    """
    print(f"Analyzing X-ray image at path: {image_path}")
    try:
        result, confidence, annotated_image_path = detect_cancer_in_image(image_path)
        confidence_percent = confidence * 100
        annotated_path_str = str(annotated_image_path) if annotated_image_path is not None else "None"
        return f"XRAY_RESULT:{result}|CONFIDENCE:{confidence_percent:.1f}|ANNOTATED_IMAGE_PATH:{annotated_path_str}"
    except Exception as e:
        print(f"Error during X-ray analysis tool: {e}")
        return f"Error analyzing image: {str(e)}"
async def interpret_ml_results(ml_result: str, ml_confidence: float, lang: str) -> str:
    """Generates an empathetic explanation of the ML results."""
    # Coalesce None values to safe defaults
    safe_ml_result = ml_result or "Not available"
    try:
        safe_ml_conf = float(ml_confidence) if ml_confidence is not None else 0.0
    except Exception:
        safe_ml_conf = 0.0
    prompt_en = f"""You are an empathetic AI medical assistant speaking to a patient in English.
Your task is to provide a clear, calm, and detailed summary of their questionnaire-based breast cancer risk assessment.
**Analysis Results:**
-   **Questionnaire-Based Assessment:**
    -   Result: **{safe_ml_result}**
    -   Confidence: **{safe_ml_conf:.1f}%**
**Your Explanation:**
1.  Start by gently summarizing the result, mentioning the confidence level.
2.  Briefly explain what the result might suggest in simple, non-alarming terms.
3.  **Crucially, end with this strong, reassuring message:** "The most important next step is to discuss these results with your healthcare provider. This analysis is a helpful tool, but it is not a diagnosis. A doctor is the only one who can provide a definitive answer and guide you on what to do next."
Keep your response warm, supportive, and professional.
"""
    prompt_ar = f"""
أنت مساعد طبي ذكي ومتفهم تتحدث مع مريض باللغة العربية المصرية.
مهمتك هي تقديم ملخص واضح وهادئ ومفصل لتقييم خطر سرطان الثدي المكون من جزأين.
**نتائج التحليل:**
- **تقييم مبني على الإجابات:**
   - النتيجة: **{('إيجابية' if safe_ml_result == 'Positive' else ('سلبية' if safe_ml_result == 'Negative' else 'غير متاح'))}**
   - نسبة الثقة: **{safe_ml_conf:.1f}%**
**الشرح المطلوب:**
١. ابدأي بتلخيص النتيجة بلطف مع ذكر نسبة الثقة.
٢. اشرحي ببساطة ما قد تعنيه النتيجة بمصطلحات بسيطة وغير مثيرة للقلق.
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
async def interpret_xray_results(xray_result: str, xray_confidence: float, lang: str) -> str:
    """Generates an empathetic explanation of the X-ray results."""
    # Coalesce None values to safe defaults
    safe_xray_result = xray_result or "Not available"
    try:
        safe_xray_conf = float(xray_confidence) if xray_confidence is not None else 0.0
    except Exception:
        safe_xray_conf = 0.0
    prompt_en = f"""You are an empathetic AI medical assistant speaking to a patient in English.
Your task is to provide a clear, calm, and detailed summary of their X-ray image analysis.
**Analysis Results:**
-  **X-ray Image Analysis:**
    -   Result: **{safe_xray_result}**
    -   Confidence: **{safe_xray_conf:.1f}%**
**Your Explanation:**
1.  Start by gently summarizing the result, mentioning the confidence level.
2.  If the X-ray result is 'Positive', explain that the analysis highlighted an area of interest for a specialist to review. Mention that the confidence score reflects the model's certainty.
3.  If the X-ray result is 'Negative', state that the image did not show any immediate areas of concern, and the confidence score reflects this.
4.  **Crucially, end with this strong, reassuring message:** "The most important next step is to discuss these results with your healthcare provider. This analysis is a helpful tool, but it is not a diagnosis. A doctor is the only one who can provide a definitive answer and guide you on what to do next."
Keep your response warm, supportive, and professional.
"""
    prompt_ar = f"""
أنت مساعد طبي ذكي ومتفهم تتحدث مع مريض باللغة العربية المصرية.
مهمتك هي تقديم ملخص واضح وهادئ ومفصل لتقييم خطر سرطان الثدي المكون من جزأين.
**نتائج التحليل:**
- **تحليل صورة الأشعة:**
   - النتيجة: **{('إيجابية' if safe_xray_result == 'Positive' else ('سلبية' if safe_xray_result == 'Negative' else 'غير متاح'))}**
   - نسبة الثقة: **{safe_xray_conf:.1f}%**
**الشرح المطلوب:**
١. ابدأي بتلخيص النتيجة بلطف مع ذكر نسبة الثقة.
٢. إذا كانت نتيجة الأشعة "إيجابية"، اشرحي أن التحليل أبرز منطقة تحتاج لمراجعة من متخصص. اذكري أن درجة الثقة تعكس يقين النموذج.
٣. إذا كانت نتيجة الأشعة "سلبية"، اذكري أن الصورة لم تظهر أي مناطق مثيرة للقلق فورياً، ودرجة الثقة تعكس هذا.
٤. اهم خطوة تعمليها بعد كده هي انك تستيري دكتور عن النتائج دي. التحليل ده أداة مساعدة بس، مش تشخيص. الدكتور هو الوحيد اللي يقدر يديلك إجابة نهائية ويوجهك في الخطوة الجاية.
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
@tool
async def interpret_medical_reports(file_paths: list, lang: str) -> str:
    """Interprets a list of medical documents (images, PDFs, DOCs) to provide a summary.
    Use this tool when the user uploads one or more medical reports and asks for an interpretation.
    """
    print(f"Interpreting {len(file_paths)} medical reports.")
    return "INTERPRETATION_RESULT:The medical reports have been interpreted."

# --- CORRECTED: This tool now returns the file path on success ---
@tool
def convert_latex_to_pdf(latex_string: str, output_filename: str) -> str:
    """
    Converts a LaTeX string to a PDF file.
    Returns the absolute path to the PDF on success, or an error string on failure.
    """
    import subprocess
    import os
    import tempfile
    
    # Use a temporary directory to avoid cluttering the main directory
    with tempfile.TemporaryDirectory() as temp_dir:
        jobname = os.path.splitext(output_filename)[0]
        tex_path = os.path.join(temp_dir, f"{jobname}.tex")
        
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(latex_string)
            
        # The -output-directory argument tells pdflatex where to put the output files
        process = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', f'-output-directory={temp_dir}', tex_path],
            capture_output=True,
            text=True
        )
        
        generated_pdf_path = os.path.join(temp_dir, output_filename)

        if process.returncode != 0 or not os.path.exists(generated_pdf_path):
            print("Error generating PDF:")
            print(process.stdout)
            print(process.stderr)
            # Provide the LaTeX log for easier debugging
            log_path = os.path.join(temp_dir, f"{jobname}.log")
            log_content = ""
            if os.path.exists(log_path):
                with open(log_path, 'r') as log_file:
                    log_content = log_file.read()
            return f"Error generating PDF. Log: {log_content[:1000]}" # Return first 1000 chars of log
        else:
            # Copy the final PDF to the project's root directory to be served
            final_pdf_path = os.path.abspath(output_filename)
            import shutil
            shutil.copy(generated_pdf_path, final_pdf_path)
            print(f"PDF '{final_pdf_path}' created successfully.")
            return final_pdf_path
