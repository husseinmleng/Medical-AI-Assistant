
import os
import base64
import markdown
from datetime import datetime

def extract_ml_assessment_from_conversation(conversation: list) -> tuple:
    """
    Extracts ML assessment results from conversation messages by looking for the technical line.
    Returns (ml_result, ml_confidence) tuple.
    """
    ml_result = None
    ml_confidence = None
    
    print(f"🔍 Extracting ML assessment from {len(conversation)} conversation messages")
    
    for msg in conversation:
        if msg.get("role") in ["assistant", "ai"]:  # Handle both role formats
            content = str(msg.get("content", ""))
            if "INITIAL_ASSESSMENT_RESULT:" in content:
                try:
                    # Find the segment after the marker
                    tech_start = content.find("INITIAL_ASSESSMENT_RESULT:")
                    tech_segment = content[tech_start:]
                    # Stop at newline if any
                    tech_segment = tech_segment.splitlines()[0]
                    # Parse key-value pairs separated by '|'
                    parts = {p.split(':', 1)[0]: p.split(':', 1)[1] for p in tech_segment.split('|') if ':' in p}
                    ml_result = parts.get("INITIAL_ASSESSMENT_RESULT")
                    conf_str = parts.get("CONFIDENCE")
                    if conf_str is not None:
                        ml_confidence = float(conf_str)
                    break
                except Exception:
                    continue
    
    # Fallback: Try to extract from Arabic text patterns if no technical line found
    if ml_result is None:
        print("🔍 No technical line found, trying Arabic text patterns...")
        for msg in conversation:
            if msg.get("role") in ["assistant", "ai"]:  # Handle both role formats
                content = str(msg.get("content", ""))
                print(f"🔍 Checking message: {content[:100]}...")
                
                # Look for Arabic patterns like "النتيجة سلبية" or "النتيجة إيجابية"
                print(f"🔍 Checking content for Arabic patterns: {content[:200]}...")
                
                # Check for negative results
                if any(pattern in content for pattern in ["النتيجة سلبية", "النتيجة: سلبية", "سلبية", "سلبى", "سلبية"]):
                    ml_result = "Negative"
                    print("✅ Found Negative result in Arabic text")
                # Check for positive results  
                elif any(pattern in content for pattern in ["النتيجة إيجابية", "النتيجة: إيجابية", "إيجابية", "إيجابى", "إيجابية"]):
                    ml_result = "Positive"
                    print("✅ Found Positive result in Arabic text")
                # Check for English results
                elif "Negative" in content:
                    ml_result = "Negative"
                    print("✅ Found Negative result in English text")
                elif "Positive" in content:
                    ml_result = "Positive"
                    print("✅ Found Positive result in English text")
                else:
                    print("❌ No result pattern found in content")
                
                # Try to extract confidence from Arabic text
                import re
                print(f"🔍 Trying to extract confidence from: {content}")
                
                # Look for percentage patterns
                confidence_patterns = [
                    r'الثقة:\s*(\d+)%',  # Handle "الثقة: 75%" format (most specific first)
                    r'الثقة\s+(\d+)%',   # Handle "الثقة 75%" format
                    r'الثقة.*?(\d+)%',   # Handle "الثقة anything 75%" format
                    r'(\d+)%.*الثقة',    # Handle "75% anything الثقة" format
                    r'حوالي\s*(\d+)%',   # Handle "حوالي 75%" format
                    r'(\d+)%'            # Fallback to any percentage
                ]
                
                for pattern in confidence_patterns:
                    print(f"🔍 Testing pattern: {pattern}")
                    match = re.search(pattern, content)
                    if match:
                        try:
                            ml_confidence = float(match.group(1))
                            print(f"✅ Found confidence: {ml_confidence}% with pattern: {pattern}")
                            break
                        except Exception as e:
                            print(f"❌ Error parsing confidence: {e}")
                    else:
                        print(f"❌ Pattern {pattern} not found")
                
                if ml_result:
                    print(f"✅ Found result: {ml_result}")
                    break
                
        # If we still don't have confidence, try a simpler approach
        if ml_result and ml_confidence is None:
            print("🔍 Trying simple confidence extraction...")
            for msg in conversation:
                if msg.get("role") in ["assistant", "ai"]:  # Handle both role formats
                    content = str(msg.get("content", ""))
                    # Look for any percentage number
                    import re
                    simple_match = re.search(r'(\d+)%', content)
                    if simple_match:
                        try:
                            ml_confidence = float(simple_match.group(1))
                            print(f"✅ Found confidence with simple extraction: {ml_confidence}%")
                            break
                        except Exception as e:
                            print(f"❌ Error in simple confidence extraction: {e}")
    
    print(f"🎯 Final extraction result: ml_result={ml_result}, ml_confidence={ml_confidence}")
    return ml_result, ml_confidence

def generate_html_report(conversation: list, patient_info: dict, analysis_results: dict, patient_name: str, report_title: str, lang: str = 'en') -> str:
    """
    Generates an HTML report from conversation data and analysis results.
    """
    # --- Extract ML Assessment Results from Conversation ---
    ml_result, ml_confidence = extract_ml_assessment_from_conversation(conversation)
    
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
        role = msg.get("role", "unknown").title()
        content = markdown.markdown(str(msg.get("content", "")))
        bubble_class = "user-bubble" if msg.get("role") in ["user", "human"] else "assistant-bubble"
        # Determine direction based on language of the message
        direction = "rtl" if lang == "ar" else "ltr"
        conversation_html.append(f'<div class="chat-bubble {bubble_class}" style="direction: {direction};"><strong>{role}:</strong>{content}</div>')

    # --- Analysis Results Formatting ---
    analysis_results_html = []
    for key, value in analysis_results.items():
        if value and key not in ['annotated_image_path', 'x-ray prediction', 'x-ray prediction confidence', 'interpretation']:
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
            <tr><th>Patient Name</th><td>Jehan Metwally</td></tr>
            {"".join([f"<tr><th>{key.replace('_', ' ').title()}</th><td>{value}</td></tr>" for key, value in patient_info.items()])}
        </table>
    </div>

    <div class="section">
        <h2>X-Ray Analysis Results</h2>
        <table class="info-table">
            <tr><th>X-Ray Result</th><td>{analysis_results.get('xray_result', 'N/A')}</td></tr>
            <tr><th>X-Ray Confidence</th><td>{analysis_results.get('xray_confidence', 'N/A')}</td></tr>
        </table>
    </div>

            <div class="section">
        <h2>Breast Cancer ML Assessment</h2>
        <table class="info-table">
            <tr><th>Assessment Result</th><td>{ml_result or 'N/A'}</td></tr>
            <tr><th>Confidence</th><td>{f'{ml_confidence:.1f}%' if ml_confidence is not None else 'N/A'}</td></tr>
        </table>
    </div>

    {annotated_image_html}

    <div class="section">
        <h2>📄 Medical Reports Interpretation</h2>
        <div style="background-color: #f0f8ff; padding: 20px; border-radius: 8px; border-left: 5px solid #007bff; margin-bottom: 15px;">
            <h3 style="color: #0056b3; margin-top: 0;">AI Analysis Results:</h3>
            <div style="background-color: white; padding: 15px; border-radius: 5px; border: 1px solid #e3f2fd;">
                {markdown.markdown(str(analysis_results.get('interpretation_result', 'No medical reports have been uploaded and analyzed yet.')))}
            </div>
        </div>
        
        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745; margin-top: 15px;">
            <h4 style="color: #2e7d32; margin-top: 0;">Uploaded Reports Summary:</h4>
            <div style="background-color: white; padding: 10px; border-radius: 3px; font-family: 'Courier New', monospace; font-size: 0.85em; max-height: 300px; overflow-y: auto; border: 1px solid #c8e6c9;">
                {analysis_results.get('reports_context', 'No medical reports have been uploaded.')}
            </div>
        </div>
    </div>

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
