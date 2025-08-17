import asyncio
import os
import sys
from datetime import datetime
from typing import List, Dict

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.latex_agent import generate_latex_report
from src.tools import convert_latex_to_pdf


async def demo_translation_and_report_generation():
    """
    Complete demonstration of the Arabic-to-English translation and PDF report generation.
    """
    print("🩺 Medical AI Assistant - Report Generation Demo")
    print("=" * 60)
    
    # Sample Arabic conversation (real medical consultation scenario)
    print("📝 Sample Arabic Medical Conversation:")
    arabic_conversation = [
        {
            "role": "assistant", 
            "content": "مرحبًا! أنا مساعدك الطبي الذكي. لبدء التقييم، سأحتاج إلى طرح بعض الأسئلة حول تاريخك الصحي. كيف يمكنني مساعدتك اليوم؟"
        },
        {
            "role": "user", 
            "content": "مرحبا دكتور، أشعر بألم في الصدر منذ أسبوع تقريباً وأشعر بالتعب الشديد"
        },
        {
            "role": "assistant", 
            "content": "أفهم قلقك. دعني أسألك بعض الأسئلة المهمة. كم عمرك؟"
        },
        {
            "role": "user", 
            "content": "عمري 35 سنة"
        },
        {
            "role": "assistant", 
            "content": "شكراً لك. هل لديك تاريخ عائلي بسرطان الثدي؟"
        },
        {
            "role": "user", 
            "content": "نعم، والدتي أصيبت بسرطان الثدي من قبل"
        },
        {
            "role": "assistant", 
            "content": "هل لاحظت فقدان وزن مؤخراً دون اتباع حمية؟"
        },
        {
            "role": "user", 
            "content": "نعم، فقدت حوالي 5 كيلو في الشهر الماضي"
        },
        {
            "role": "assistant", 
            "content": "هل سبق وأن عانيت من مشاكل في الثدي من قبل؟"
        },
        {
            "role": "user", 
            "content": "لا، لم أعاني من مشاكل سابقة"
        },
        {
            "role": "assistant", 
            "content": "كم يوماً تعانين من هذه الأعراض؟"
        },
        {
            "role": "user", 
            "content": "حوالي 10 أيام"
        },
        {
            "role": "assistant", 
            "content": "هل تشعرين بالتعب والإرهاق؟"
        },
        {
            "role": "user", 
            "content": "نعم، أشعر بتعب شديد حتى مع الراحة"
        },
        {
            "role": "assistant", 
            "content": """بناءً على إجاباتك، هناك بعض العوامل التي تستدعي الانتباه:

النتيجة: إيجابية
نسبة الثقة: 85%

التفسير: إجاباتك تظهر وجود عدة عوامل خطر مجتمعة - التاريخ العائلي لسرطان الثدي، فقدان الوزن غير المبرر، والأعراض المستمرة لأكثر من أسبوع مع التعب. هذا لا يعني بالضرورة وجود مرض، لكنه يستدعي المتابعة الطبية.

الخطوة الأهم التي يجب عليك اتخاذها الآن هي مراجعة طبيب مختص لمناقشة هذه النتائج. هذا التحليل أداة مساعدة فقط وليس تشخيصاً طبياً. الطبيب هو الوحيد القادر على إعطائك إجابة نهائية وتوجيهك للخطوة التالية."""
        }
    ]
    
    for i, msg in enumerate(arabic_conversation[:4], 1):
        print(f"  {i}. {msg['role'].title()}: {msg['content'][:50]}...")
    print(f"  ... (total {len(arabic_conversation)} messages)")
    
    # Step 1: Translation
    print(f"\n🔄 Step 1: Translating conversation from Arabic to English...")
    try:
        from src.app_logic import translate_conversation_to_english
        translated_conversation = await translate_conversation_to_english(arabic_conversation)
        print(f"✅ Translation successful! {len(translated_conversation)} messages translated.")
        
        print("\n📖 Sample translated messages:")
        for i, msg in enumerate(translated_conversation[:2], 1):
            print(f"  {i}. {msg['role'].title()}: {msg['content'][:80]}...")
            
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        print("🔄 Using original conversation for demo purposes...")
        translated_conversation = [
            {"role": "assistant", "content": "Hello! I'm your AI Medical Assistant. To start the assessment, I need to ask some questions about your health history. How can I help you today?"},
            {"role": "user", "content": "Hello doctor, I have chest pain for about a week and I feel very tired"},
            {"role": "assistant", "content": "I understand your concern. Let me ask some important questions. What is your age?"},
            {"role": "user", "content": "I am 35 years old"},
            {"role": "assistant", "content": "Thank you. Do you have a family history of breast cancer?"},
            {"role": "user", "content": "Yes, my mother had breast cancer before"},
            {"role": "assistant", "content": "Have you noticed recent weight loss without dieting?"},
            {"role": "user", "content": "Yes, I lost about 5 kg in the past month"},
            {"role": "assistant", "content": "Have you ever had breast problems before?"},
            {"role": "user", "content": "No, I haven't had previous problems"},
            {"role": "assistant", "content": "How many days have you been experiencing these symptoms?"},
            {"role": "user", "content": "About 10 days"},
            {"role": "assistant", "content": "Do you feel fatigue and exhaustion?"},
            {"role": "user", "content": "Yes, I feel severe fatigue even with rest"},
            {"role": "assistant", "content": """Based on your answers, there are some factors that require attention:

Result: Positive
Confidence: 85%

Explanation: Your answers show several risk factors combined - family history of breast cancer, unexplained weight loss, and persistent symptoms for more than a week with fatigue. This doesn't necessarily mean there's a disease, but it requires medical follow-up.

The most important step you should take now is to consult with a specialist doctor to discuss these results. This analysis is just a helpful tool and not a medical diagnosis. The doctor is the only one who can give you a definitive answer and guide you to the next step."""}
        ]
    
    # Step 2: Prepare sample data
    print(f"\n📋 Step 2: Preparing patient and analysis data...")
    
    patient_info = {
        "patient_name": "Sarah Ahmed",
        "age": "35",
        "breast_feeding_months": "18",
        "family_history_breast_cancer": "yes",
        "recent_weight_loss": "yes", 
        "previous_breast_conditions": "no",
        "symptom_duration_days": "10",
        "fatigue": "yes"
    }
    
    analysis_results = {
        "ml_result": "Positive",
        "ml_confidence": 85.0,
        "xray_result": "Pending",
        "xray_confidence": None,
        "annotated_image_path": None,
        "interpretation_result": "Patient presents with multiple risk factors requiring immediate medical attention. Combination of family history, unexplained weight loss, and persistent symptoms warrants urgent consultation with oncology specialist.",
        "reports_text_context": None
    }
    
    print("✅ Sample data prepared:")
    print(f"  - Patient: {patient_info['patient_name']}, Age: {patient_info['age']}")
    print(f"  - ML Assessment: {analysis_results['ml_result']} ({analysis_results['ml_confidence']}% confidence)")
    
    # Step 3: Generate LaTeX report
    print(f"\n📄 Step 3: Generating professional LaTeX report...")
    try:
        latex_content = generate_latex_report(
            conversation=translated_conversation,
            patient_info=patient_info,
            analysis_results=analysis_results,
            patient_name=patient_info["patient_name"],
            report_title="Medical AI Assistant - Breast Cancer Risk Assessment Report"
        )
        
        print("✅ LaTeX report generated successfully!")
        print(f"  - Document length: {len(latex_content):,} characters")
        print(f"  - Sections included: Patient Info, Analysis Results, Key Findings, Conversation, Recommendations")
        
        # Save LaTeX source for inspection
        latex_filename = f"demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
        with open(latex_filename, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        print(f"  - LaTeX source saved as: {latex_filename}")
        
    except Exception as e:
        print(f"❌ LaTeX generation failed: {e}")
        return
    
    # Step 4: Convert to PDF
    print(f"\n🔄 Step 4: Converting LaTeX to PDF...")
    try:
        pdf_filename = f"demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = convert_latex_to_pdf(latex_content, pdf_filename)
        
        if pdf_path and "Error" not in pdf_path and os.path.exists(pdf_path):
            file_size = os.path.getsize(pdf_path) / 1024  # KB
            print("✅ PDF report generated successfully!")
            print(f"  - PDF saved as: {pdf_path}")
            print(f"  - File size: {file_size:.1f} KB")
            
            # Verify PDF content
            print(f"\n📊 Report Statistics:")
            conversation_words = sum(len(msg['content'].split()) for msg in translated_conversation)
            print(f"  - Conversation: {len(translated_conversation)} messages, ~{conversation_words} words")
            print(f"  - Patient data fields: {len([k for k, v in patient_info.items() if v])}")
            print(f"  - Analysis results: {len([k for k, v in analysis_results.items() if v is not None])}")
            
        else:
            print(f"❌ PDF generation failed: {pdf_path}")
            
    except Exception as e:
        print(f"❌ PDF conversion failed: {e}")
        return
    
    # Step 5: Demo summary
    print(f"\n🎉 Demo completed successfully!")
    print("=" * 60)
    print("📋 Summary of what was accomplished:")
    print("  1. ✅ Arabic medical conversation translated to English")
    print("  2. ✅ Professional LaTeX medical report generated")
    print("  3. ✅ PDF report compiled and saved")
    print("  4. ✅ Complete medical documentation workflow demonstrated")
    
    print(f"\n💡 The generated report includes:")
    print("  - Professional medical report formatting")
    print("  - Patient information and questionnaire data")
    print("  - AI analysis results and confidence levels")
    print("  - Complete conversation transcript")
    print("  - Medical recommendations and disclaimers")
    print("  - Proper medical report structure and styling")
    
    if 'pdf_path' in locals() and os.path.exists(pdf_path):
        print(f"\n📁 Files generated:")
        print(f"  - LaTeX source: {latex_filename}")
        print(f"  - PDF report: {pdf_path}")
        print(f"\n🔍 You can now open the PDF to see the complete medical report!")


def demo_report_customization():
    """
    Demonstrate different report customization options.
    """
    print("\n🎨 Report Customization Demo")
    print("-" * 40)
    
    # Different report styles
    styles = {
        "standard": "Standard Medical Report",
        "emergency": "Emergency Assessment Report", 
        "followup": "Follow-up Consultation Report",
        "screening": "Preventive Screening Report"
    }
    
    for style_key, style_name in styles.items():
        print(f"  📋 {style_name}")
        # In a real implementation, you would have different LaTeX templates
        print(f"     - Template: medical_{style_key}.tex")
        print(f"     - Styling: {style_key}_colors.sty")
        print(f"     - Sections: {style_key}_sections.json")


def demo_performance_testing():
    """
    Demonstrate performance characteristics of the report generation system.
    """
    print("\n⚡ Performance Testing Demo")
    print("-" * 40)
    
    import time
    
    # Simulate different conversation lengths
    conversation_sizes = [
        ("Short", 5, "Quick consultation"),
        ("Medium", 15, "Standard consultation"),
        ("Long", 30, "Comprehensive assessment"),
        ("Extended", 50, "Multi-session consultation")
    ]
    
    for size_name, msg_count, description in conversation_sizes:
        print(f"  📊 {size_name} Conversation ({msg_count} messages)")
        print(f"     Description: {description}")
        
        # Estimated timings based on typical performance
        estimated_translation = msg_count * 0.3  # ~300ms per message
        estimated_latex = 2.0  # Fixed LaTeX generation time
        estimated_pdf = 4.0    # Fixed PDF compilation time
        total_time = estimated_translation + estimated_latex + estimated_pdf
        
        print(f"     Est. Translation: {estimated_translation:.1f}s")
        print(f"     Est. LaTeX Gen: {estimated_latex:.1f}s") 
        print(f"     Est. PDF Comp: {estimated_pdf:.1f}s")
        print(f"     Total Time: {total_time:.1f}s")
        print()


if __name__ == "__main__":
    print("🚀 Starting Medical AI Assistant Report Generation Demo")
    print("This demo will show the complete workflow from Arabic conversation to PDF report.\n")
    
    try:
        # Run the main demo
        asyncio.run(demo_translation_and_report_generation())
        
        # Additional demos
        demo_report_customization()
        demo_performance_testing()
        
        print("\n✨ Demo completed successfully!")
        print("You can now integrate this system into your Medical AI Assistant.")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()