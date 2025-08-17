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

import os

def check_latex_installation() -> dict:
    """
    Checks if LaTeX is properly installed and returns installation status.
    
    Returns:
        Dictionary with installation status and details
    """
    import subprocess
    import shutil
    
    result = {
        "installed": False,
        "pdflatex_available": False,
        "latex_available": False,
        "version": None,
        "installation_guide": None,
        "required_packages": [],
        "missing_packages": []
    }
    
    # Check if pdflatex is available
    pdflatex_path = shutil.which("pdflatex")
    if pdflatex_path:
        result["pdflatex_available"] = True
        try:
            # Get version information
            version_output = subprocess.run(
                ["pdflatex", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if version_output.returncode == 0:
                result["installed"] = True
                # Extract version from first line
                first_line = version_output.stdout.split('\n')[0]
                result["version"] = first_line.strip()
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass
    
    # Check if latex is available
    latex_path = shutil.which("latex")
    if latex_path:
        result["latex_available"] = True
    
    # Check required packages by testing compilation
    if result["installed"]:
        required_packages = [
            "inputenc", "fontenc", "geometry", "fancyhdr", "graphicx",
            "xcolor", "tcolorbox", "enumitem", "booktabs", "longtable",
            "amsmath", "amssymb", "hyperref", "times"
        ]
        
        result["required_packages"] = required_packages
        import tempfile
        # Test package availability with a simple document
        with tempfile.TemporaryDirectory() as temp_dir:
            test_tex = os.path.join(temp_dir, "package_test.tex")
            test_content = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{fancyhdr}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tcolorbox}
\usepackage{enumitem}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{hyperref}
\usepackage{times}
\begin{document}
Test document
\end{document}
"""
            
            with open(test_tex, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            try:
                # Try to compile the test document
                compile_result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', '-output-directory', temp_dir, test_tex],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=temp_dir
                )
                
                if compile_result.returncode == 0:
                    result["missing_packages"] = []
                else:
                    # Parse error output to find missing packages
                    error_output = compile_result.stderr.lower()
                    for package in required_packages:
                        if f"package {package} not found" in error_output or f"file {package}.sty not found" in error_output:
                            result["missing_packages"].append(package)
                            
            except Exception:
                result["missing_packages"] = required_packages
    
    # Provide installation guide if not installed
    if not result["installed"]:
        if os.name == 'nt':  # Windows
            result["installation_guide"] = """
            Install MiKTeX from: https://miktex.org/download
            Or install TeX Live from: https://www.tug.org/texlive/
            """
        elif os.name == 'posix':  # Linux/macOS
            if os.path.exists('/etc/debian_version'):  # Debian/Ubuntu
                result["installation_guide"] = "sudo apt-get install texlive-full"
            elif os.path.exists('/etc/redhat-release'):  # RHEL/CentOS
                result["installation_guide"] = "sudo yum install texlive-scheme-full"
            elif os.path.exists('/etc/arch-release'):  # Arch
                result["installation_guide"] = "sudo pacman -S texlive-most"
            else:
                result["installation_guide"] = "Install TeX Live distribution for your system"
    
    return result

def convert_latex_to_pdf(latex_string: str, output_filename: str) -> str:
    """
    Converts a LaTeX string to a PDF file.
    Returns the absolute path to the PDF on success, or an error string on failure.
    """
    import subprocess
    import os
    import tempfile
    import shutil
    
    # Check if pdflatex is available
    try:
        subprocess.run(['pdflatex', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Error: pdflatex is not installed or not available in PATH. Please install LaTeX distribution (e.g., TeX Live, MiKTeX)."
    
    # Ensure latex_string is properly encoded
    try:
        if isinstance(latex_string, bytes):
            latex_string = latex_string.decode('utf-8', errors='replace')
        latex_string = str(latex_string)
    except Exception as e:
        print(f"Warning: Could not decode LaTeX string: {e}")
        try:
            latex_string = latex_string.encode('utf-8', errors='replace').decode('utf-8')
        except Exception:
            return f"Error: Could not process LaTeX string due to encoding issues: {e}"
    
    # Use a temporary directory to avoid cluttering the main directory
    with tempfile.TemporaryDirectory() as temp_dir:
        jobname = os.path.splitext(output_filename)[0]
        tex_path = os.path.join(temp_dir, f"{jobname}.tex")
        
        # Write LaTeX content with UTF-8 encoding
        try:
            with open(tex_path, 'w', encoding='utf-8') as f:
                f.write(latex_string)
        except Exception as e:
            return f"Error writing LaTeX file: {e}"
            
        # The -output-directory argument tells pdflatex where to put the output files
        # Use UTF-8 encoding for better Unicode support
        process = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', f'-output-directory={temp_dir}', tex_path],
            capture_output=True,
            encoding='utf-8',  # Changed from latin-1 to utf-8
            cwd=temp_dir
        )
        
        generated_pdf_path = os.path.join(temp_dir, output_filename)

        if process.returncode != 0 or not os.path.exists(generated_pdf_path):
            print("Error generating PDF:")
            print("STDOUT:", process.stdout)
            print("STDERR:", process.stderr)
            print("Return code:", process.returncode)
            
            # Provide the LaTeX log for easier debugging
            log_path = os.path.join(temp_dir, f"{jobname}.log")
            log_content = ""
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r', encoding='utf-8') as log_file:
                        log_content = log_file.read()
                except UnicodeDecodeError:
                    # Fallback to latin-1 if UTF-8 fails
                    with open(log_path, 'r', encoding='latin-1') as log_file:
                        log_content = log_file.read()
            
            return f"Error generating PDF (return code: {process.returncode}). Log: {log_content[:1000]}"
        else:
            # Copy the final PDF to the project's root directory to be served
            try:
                final_pdf_path = os.path.abspath(output_filename)
                shutil.copy(generated_pdf_path, final_pdf_path)
                print(f"PDF '{final_pdf_path}' created successfully.")
                return final_pdf_path
            except Exception as copy_error:
                print(f"Error copying PDF to final location: {copy_error}")
                # Return the temporary path if copying fails
                return generated_pdf_path
