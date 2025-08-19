
import os
import base64
import markdown
from datetime import datetime

def generate_html_report(conversation: list, patient_info: dict, analysis_results: dict, patient_name: str, report_title: str, lang: str = 'en') -> str:
    """
    Generates an HTML report from conversation data and analysis results.
    """
    # --- Image Embedding ---
    annotated_image_html = ""
    annotated_image_path = analysis_results.get("annotated_image_path")
    if annotated_image_path and os.path.exists(annotated_image_path):
        try:
            with open(annotated_image_path, "rb") as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode("utf-8")
                annotated_image_html = f'''
    <div class="section">
        <h2>Annotated X-Ray Image</h2>
        <img src="data:image/png;base64,{img_base64}" alt="Annotated X-Ray" style="max-width: 100%; height: auto;">
    </div>
'''
        except Exception as e:
            print(f"Error embedding image: {e}")
            annotated_image_html = "<p>Error loading annotated image.</p>"

    # --- Conversation Formatting ---
    conversation_html = []
    for msg in conversation:
        role = msg["role"].title()
        content = markdown.markdown(msg["content"])
        bubble_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
        # Determine direction based on language of the message
        direction = "rtl" if lang == "ar" else "ltr"
        conversation_html.append(f'<div class="chat-bubble {bubble_class}" style="direction: {direction};"><strong>{role}:</strong>{content}</div>')

    # --- Analysis Results Formatting ---
    analysis_results_html = []
    for key, value in analysis_results.items():
        if value and key not in ['annotated_image_path', 'prediction', 'confidence', 'interpretation']:
            analysis_results_html.append(f"<tr><th>{key.replace('_', ' ').title()}</th><td>{value}</td></tr>")

    # Basic HTML structure and styling
    html_template = f"""
<!DOCTYPE html>
<html lang="{lang}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        .header h1 {{
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #444;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
            margin-bottom: 15px;
        }}
        .chat-bubble {{
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 10px;
        }}
        .user-bubble {{
            background-color: #f1f1f1;
            text-align: left;
        }}
        .assistant-bubble {{
            background-color: #e2f0ff;
            text-align: left;
        }}
        .info-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .info-table th, .info-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        .info-table th {{
            background-color: #f2f2f2;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            font-size: 0.8em;
            color: #777;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report_title}</h1>
        <p>Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="section">
        <h2>Patient Information</h2>
        <table class="info-table">
            <tr><th>Patient Name</th><td>{patient_name}</td></tr>
            {"".join([f"<tr><th>{key.replace('_', ' ').title()}</th><td>{value}</td></tr>" for key, value in patient_info.items()])}
        </table>
    </div>

    <div class="section">
        <h2>Analysis Results</h2>
        <table class="info-table">
            {"".join(analysis_results_html)}
        </table>
    </div>

    <div class="section">
        <h2>Breast Cancer ML Prediction</h2>
        <table class="info-table">
            <tr><th>Prediction</th><td>{analysis_results.get('prediction', 'N/A')}</td></tr>
            <tr><th>Confidence</th><td>{analysis_results.get('confidence', 'N/A')}</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>Report Interpretation</h2>
        <p>{analysis_results.get('interpretation', 'N/A')}</p>
    </div>

    {annotated_image_html}

    <div class="section">
        <h2>Conversation Transcript</h2>
        {"".join(conversation_html)}
    </div>

    <div class="footer">
        <p>This report was generated automatically. Always consult with a qualified medical professional for diagnosis.</p>
    </div>
</body>
</html>
"""
    return html_template
