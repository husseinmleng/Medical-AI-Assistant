from typing import List, Dict, Any
import datetime
import os
import base64
import mimetypes

def _image_to_latex_include(image_path: str, width: str = "0.8\textwidth") -> str:
    """Converts an image file to a base64 string and includes it in LaTeX."""
    if not image_path or not os.path.exists(image_path):
        return ""
    
    # For simplicity, we'll assume images are already in a format LaTeX can handle
    # or that they will be converted externally. Here, we just include the path.
    # In a real scenario, you might copy the image to a temp dir and reference it.
    # Or, if using pdflatex, you might need to convert to .eps or .pdf first.
    # For now, we'll just include the path directly, assuming it's accessible.
    
    # If we want to embed base64, it's more complex and usually requires a package like `graphicx`
    # and potentially a custom LaTeX command or external processing.
    # For direct LaTeX compilation, it's usually better to reference a file.
    
    # Let's assume the image is a PNG/JPG and can be directly included by pdflatex
    # if it's in the same directory or a known path.
    # We'll use a relative path from where pdflatex is run, or an absolute path.
    # Given the Streamlit app, the annotated images are in 'annotated_images' folder
    # which is at the project root.
    
    # To make it robust, we should ensure the image path is relative to the LaTeX compilation directory
    # or copy it there. For now, let's assume the path is directly usable.
    
    # Example: /media/husseinmleng/New Volume/Jupyter_Notebooks/Freelancing/Breast-Cancer/annotated_images/some_image.png
    # We need to make sure pdflatex can find this. 
    
    # For now, let's just return the LaTeX include command.
    # The user will need to ensure the image is accessible to pdflatex.
    return f"\\includegraphics[width={width}]{{{image_path}}}"


def generate_latex_report(
    conversation: List[Dict[str, Any]],
    patient_info: Dict[str, Any],
    analysis_results: Dict[str, Any],
    report_datetime: datetime.datetime = None,
    patient_name: str = "Patient Name Not Provided",
    report_title: str = "Medical Analysis Report"
) -> str:
    """
    Generates a LaTeX string for a comprehensive medical report.

    Args:
        conversation: List of chat messages (role, content).
        patient_info: Dictionary of questionnaire inputs (e.g., age, family_history_breast_cancer).
        analysis_results: Dictionary containing ML and X-ray analysis results.
        report_datetime: Optional datetime object for the report. Defaults to now.
        patient_name: Optional string for patient's name.
        report_title: Optional string for the report title.
    Returns:
        A string containing the LaTeX code for the report.
    """
    if report_datetime is None:
        report_datetime = datetime.datetime.now()

    latex_content = []

    # Document preamble
    latex_content.append(r"\documentclass[12pt]{article}")
    latex_content.append(r"\usepackage[utf8]{inputenc}")
    latex_content.append(r"\usepackage[T1]{fontenc}")
    latex_content.append(r"\usepackage{amsmath}")
    latex_content.append(r"\usepackage{amsfonts}")
    latex_content.append(r"\usepackage{amssymb}")
    latex_content.append(r"\usepackage{graphicx}") # For including images
    latex_content.append(r"\usepackage{caption}") # For \captionof
    latex_content.append(r"\usepackage[margin=1in]{geometry}")
    latex_content.append(r"\usepackage{fancyhdr}")
    latex_content.append(r"\pagestyle{fancy}")
    latex_content.append(r"\fancyhf{}")
    latex_content.append(r"\rhead{\thepage}")
    latex_content.append(r"\lhead{" + report_title + r"}")
    latex_content.append(r"\renewcommand{\headrulewidth}{0.4pt}")
    latex_content.append(r"\renewcommand{\footrulewidth}{0pt}")
    latex_content.append(r"\begin{document}")
    latex_content.append(r"\begin{center}")
    latex_content.append(r"\Huge\textbf{" + report_title + r"}\\[10pt]")
    latex_content.append(r"\large Date: " + report_datetime.strftime("%Y-%m-%d %H:%M:%S") + r"\\[10pt]")
    latex_content.append(r"\end{center}")
    latex_content.append(r"\hrule")
    latex_content.append(r"\section*{Patient Information}")
    latex_content.append(r"\textbf{Patient Name:} " + patient_name + r"\\")
    
    # Questionnaire Results
    if patient_info:
        latex_content.append(r"\subsection*{Questionnaire Results}")
        latex_content.append(r"\begin{itemize}")
        for key, value in patient_info.items():
            # Format keys for readability
            formatted_key = key.replace('_', ' ').title()
            latex_content.append(f"    \item \textbf{{{formatted_key}:}} {value}")
        latex_content.append(r"\end{itemize}")

    # ML Analysis Results
    ml_result = analysis_results.get("ml_result")
    ml_confidence = analysis_results.get("ml_confidence")
    if ml_result:
        latex_content.append(r"\section*{Machine Learning Analysis}")
        latex_content.append(r"\textbf{Result:} " + str(ml_result) + r"\\")
        latex_content.append(r"\textbf{Confidence:} " + (f"{ml_confidence:.1f}\%" if ml_confidence is not None else "N/A") + r"\\")
        # Add a general statement about the nature of ML analysis
        latex_content.append(r"\textit{This analysis is based on a machine learning model trained on questionnaire data. It provides a risk assessment and is not a definitive diagnosis.}")

    # X-ray Analysis Results
    xray_result = analysis_results.get("xray_result")
    xray_confidence = analysis_results.get("xray_confidence")
    annotated_image_path = analysis_results.get("annotated_image_path")
    if xray_result:
        latex_content.append(r"\section*{X-ray Image Analysis}")
        latex_content.append(r"\textbf{Result:} " + str(xray_result) + r"\\")
        latex_content.append(r"\textbf{Confidence:} " + (f"{xray_confidence:.1f}\%" if xray_confidence is not None else "N/A") + r"\\")
        if annotated_image_path:
            latex_content.append(r"\subsection*{Annotated X-ray Image}")
            latex_content.append(r"\begin{center}")
            latex_content.append(_image_to_latex_include(annotated_image_path))
            latex_content.append(r"\captionof{figure}{Annotated X-ray Image}")
            latex_content.append(r"\end{center}")
        latex_content.append(r"\textit{This analysis is based on an AI model's interpretation of the provided X-ray image. It highlights areas of interest and is not a definitive diagnosis.}")

    # Medical Report Interpretation
    interpretation_result = analysis_results.get("interpretation_result")
    reports_text_context = analysis_results.get("reports_text_context")
    if interpretation_result or reports_text_context:
        latex_content.append(r"\section*{Medical Report Interpretation}")
        if interpretation_result:
            latex_content.append(r"\subsection*{Summary of Reports}")
            # Escape special LaTeX characters in the interpretation result
            escaped_interpretation = interpretation_result.replace('&', '\\&').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#').replace('_', '\\_').replace('{', '\\{').replace('}', '\\}').replace('~', '\\textasciitilde{}').replace('^', '\\textasciicircum{}').replace('\\', '\\textbackslash{}')
            latex_content.append(escaped_interpretation + r"\\")
        if reports_text_context:
            latex_content.append(r"\subsection*{Raw Report Context (for reference)}")
            latex_content.append(r"\begin{verbatim}")
            latex_content.append(reports_text_context)
            latex_content.append(r"\end{verbatim}")
        latex_content.append(r"\textit{This section provides an AI-generated summary and context from uploaded medical documents. Always consult a medical professional for detailed understanding and diagnosis.}")

    # Conversation History
    if conversation:
        latex_content.append(r"\section*{Conversation History}")
        latex_content.append(r"\begin{itemize}")
        for msg in conversation:
            role = msg.get("role", "unknown").replace('_', ' ').title()
            content = msg.get("content", "")
            # Filter out technical messages
            if "INITIAL_ASSESSMENT_RESULT:" in content or "XRAY_RESULT:" in content or "INTERPRETATION_RESULT:" in content:
                continue
            # Escape special LaTeX characters in conversation content
            escaped_content = content.replace('&', '\\&').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#').replace('_', '\\_').replace('{', '\\{').replace('}', '\\}').replace('~', '\\textasciitilde{}').replace('^', '\\textasciicircum{}').replace('\\', '\\textbackslash{}')
            latex_content.append(f"    \item \textbf{{{role}:}} {escaped_content}")
        latex_content.append(r"\end{itemize}")

    # Important Disclaimer
    latex_content.append(r"\section*{Important Disclaimer}")
    latex_content.append(r"\textit{This report is generated by an AI assistant and is intended for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read in this report.}")

    latex_content.append(r"\end{document}")

    return "\n".join(latex_content)