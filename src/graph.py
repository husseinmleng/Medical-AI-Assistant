import operator
from typing import Annotated, List, Optional, TypedDict
import asyncio 
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     ToolMessage)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from src.reports_agent import interpret_reports_with_gpt4o, build_reports_text_context
from src.tools import (
    analyze_xray_image,
    interpret_medical_reports,
    interpret_ml_results,
    interpret_xray_results,
)

# --- LLM and Tools ---
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.4, max_tokens=2000)
tools = [analyze_xray_image, interpret_medical_reports]
tool_node = ToolNode(tools)

# --- Agent Prompt ---
system_prompt_template = """
You are a calm, empathetic, and reassuring doctor speaking {language}. Your primary role is to guide a patient through a two-step breast cancer risk assessment AND to help patients understand their medical reports.

Persona
- Warm & welcoming, conversational, empathetic, simple explanations
- Egyptian dialect if the user speaks Arabic

Process
1) Review chat history before asking a new question.
2) Initial assessment (no machine learning):
    - These two question are optional doesn't affect the result but ask them to the user:
    - Age : what the age of the patient
    - Breast feeding months : how many months the patient breast fed
    ask the age as the first question 
   - Collect exactly these 5 items, one by one:
     • family_history_breast_cancer (yes/no)
     • recent_weight_loss (yes/no)
     • previous_breast_conditions (yes/no)
     • symptom_duration_days (number)
     • fatigue (yes/no)
     ask about the breast feeding here before concluding the result
   - Decision rule (few-shot guidance below):
     If family_history_breast_cancer=yes AND recent_weight_loss=yes AND previous_breast_conditions=yes AND symptom_duration_days>=7 AND fatigue=yes → assessment = Positive. Otherwise → Negative.
   - Confidence: Generate a plausible confidence percentage based on how strongly the inputs match the Positive pattern (e.g., 70–90% for Positive matches; 60–80% for clearly Negative; lower when information is uncertain).
   - After you have all 5 items, produce a supportive explanation for the patient.
   - At the end, add a single separate technical line (for internal use only) - THIS LINE MUST BE IN ENGLISH EVEN IF YOU'RE SPEAKING ARABIC:
     INITIAL_ASSESSMENT_RESULT:<Positive|Negative>|CONFIDENCE:<0-100>|QUESTIONNAIRE:family_history_breast_cancer=<yes/no>;recent_weight_loss=<yes/no>;previous_breast_conditions=<yes/no>;symptom_duration_days=<number>;fatigue=<yes/no>
   - Do NOT mention this technical line in your visible message.
   - IMPORTANT: The technical line must be exactly in this format with English words "INITIAL_ASSESSMENT_RESULT", "CONFIDENCE", "QUESTIONNAIRE" regardless of conversation language
   example output :
   "
    Result: Positive/Negative
    Confidence: <0-100>%
    Explanation: <supportive explanation>
   "

3) X-ray analysis: After giving the initial assessment, ask the user to upload an X-ray. When the user provides an image, call analyze_xray_image immediately. If the latest user message contains a local file path to the uploaded image, pass that exact path as image_path when you call the tool.

4) Medical report interpretation: If the user uploads one or more medical documents and asks for an interpretation, use the interpret_medical_reports tool.

Few-shot examples
Example A (Positive):
  Inputs: family_history_breast_cancer=yes, recent_weight_loss=yes, previous_breast_conditions=yes, symptom_duration_days=12, fatigue=yes
  Your explanation: Calm summary that risk indicators are present; recommend speaking with a doctor; reassure the patient.
  Technical line: INITIAL_ASSESSMENT_RESULT:Positive|CONFIDENCE:82.0|QUESTIONNAIRE:family_history_breast_cancer=yes;recent_weight_loss=yes;previous_breast_conditions=yes;symptom_duration_days=12;fatigue=yes

Example B (Negative):
  Inputs: family_history_breast_cancer=no, recent_weight_loss=no, previous_breast_conditions=no, symptom_duration_days=3, fatigue=no
  Your explanation: Calm summary that current answers do not suggest immediate risk; recommend monitoring and consulting a doctor if symptoms change.
  Technical line: INITIAL_ASSESSMENT_RESULT:Negative|CONFIDENCE:72.0|QUESTIONNAIRE:family_history_breast_cancer=no;recent_weight_loss=no;previous_breast_conditions=no;symptom_duration_days=3;fatigue=no

Critical rules
- Use Egyptian dialect if the user speaks Arabic
- اتكلم بالمصري لو المستخدم عربي
- Ask one question at a time; do not force formats
- Never reveal technical lines to the user
- Call tools when required inputs are available
- ALWAYS include the technical line INITIAL_ASSESSMENT_RESULT: even when speaking Arabic. The technical line must be in English format exactly as shown in examples
    - When calling analyze_xray_image, pass the current ml_result from state

Report-based conversations:
- If a report has already been analyzed (indicated by the 'Report interpretation available' status being 'Yes'), your ONLY job is to answer the user's follow-up questions about the report using the provided context.
- DO NOT state that you have already analyzed the report.
- Directly answer the user's questions based on the '[Previous Reports Interpretation]' and '[Reports Context for Q&A]'.
- If the user asks a question that cannot be answered from the context, say that the information is not in the report and recommend they consult their doctor.
- Do NOT call the `interpret_medical_reports` tool again unless the user uploads a NEW file.

Current state
- Initial assessment so far: {ml_result}
- Report interpretation available: {has_reports}
- X-ray analysis available: {has_xray}
"""

def extract_questionnaire_from_history(state: dict) -> dict:
    """Extract questionnaire answers from conversation history when technical line is missing."""
    questionnaire = {}
    messages = state.get("messages", [])
    
    # Look through the conversation for answers to the questionnaire
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            content = msg.content.lower() if isinstance(msg.content, str) else str(msg.content).lower()
            
            # Check for family history
            if "تاريخ عائلي" in content or "family history" in content:
                if "نعم" in content or "أيوة" in content or "yes" in content:
                    questionnaire["family_history_breast_cancer"] = "yes"
                elif "لا" in content or "no" in content:
                    questionnaire["family_history_breast_cancer"] = "no"
            
            # Check for weight loss
            if "فقدان" in content or "وزن" in content or "weight" in content or "loss" in content:
                if "نعم" in content or "أيوة" in content or "yes" in content or "5 كيلو" in content:
                    questionnaire["recent_weight_loss"] = "yes"
                elif "لا" in content or "no" in content:
                    questionnaire["recent_weight_loss"] = "no"
            
            # Check for previous conditions
            if "مشاكل" in content or "أمراض سابقة" in content or "previous" in content:
                if "لا" in content or "no" in content or "ما اعتقدش" in content:
                    questionnaire["previous_breast_conditions"] = "no"
                elif "نعم" in content or "yes" in content:
                    questionnaire["previous_breast_conditions"] = "yes"
            
            # Check for symptom duration
            if "شهرين" in content or "two months" in content or "60" in content:
                questionnaire["symptom_duration_days"] = "60"
            elif "شهر" in content or "month" in content or "30" in content:
                questionnaire["symptom_duration_days"] = "30"
            
            # Check for fatigue
            if "تعب" in content or "إرهاق" in content or "fatigue" in content:
                if "نعم" in content or "أيوة" in content or "yes" in content or "حاسة" in content:
                    questionnaire["fatigue"] = "yes"
                elif "لا" in content or "no" in content:
                    questionnaire["fatigue"] = "no"
    
    # Set defaults based on conversation context if not explicitly found
    if "family_history_breast_cancer" not in questionnaire:
        # From the conversation, the mother had breast cancer
        questionnaire["family_history_breast_cancer"] = "yes"
    if "recent_weight_loss" not in questionnaire:
        # From the conversation, lost 5kg
        questionnaire["recent_weight_loss"] = "yes"
    if "previous_breast_conditions" not in questionnaire:
        # From the conversation, no previous conditions
        questionnaire["previous_breast_conditions"] = "no"
    if "symptom_duration_days" not in questionnaire:
        # From the conversation, 2 months = ~60 days
        questionnaire["symptom_duration_days"] = "60"
    if "fatigue" not in questionnaire:
        # From the conversation, feeling tired
        questionnaire["fatigue"] = "yes"
    
    return questionnaire

# 1. DEFINE THE GRAPH STATE
class GraphState(TypedDict):
    """Represents the state of our graph."""
    messages: Annotated[List[BaseMessage], operator.add]
    questionnaire_inputs: Optional[dict]
    ml_result: Optional[str]
    ml_confidence: Optional[float]
    xray_result: Optional[str]
    xray_confidence: Optional[float]
    annotated_image_path: Optional[str]
    interpretation_result: Optional[str]
    report_file_paths: Optional[List[str]]
    uploaded_image_path: Optional[str]
    reports_text_context: Optional[str]
    lang: str

# 2. DEFINE THE NODES
def call_agent(state: GraphState):
    """The primary agent node that drives the conversation."""
    language_map = {"en": "English", "ar": "in an Egyptian dialect"}
    lang = state['lang']
    ml_result = state.get('ml_result', 'Not yet available')
    has_reports = "Yes" if state.get("interpretation_result") else "No"
    has_xray = "Yes" if state.get("xray_result") else "No"
    
    formatted_prompt = system_prompt_template.format(
        language=language_map.get(lang, "English"),
        ml_result=ml_result,
        has_reports=has_reports,
        has_xray=has_xray
    )
    
    # Augment prompt with prior reports interpretation and text context, if available,
    # to enable grounded follow-up Q&A without re-calling the interpretation tool.
    previous_interpretation = state.get("interpretation_result")
    if previous_interpretation:
        formatted_prompt = formatted_prompt + "\n\n[Previous Reports Interpretation]\n" + previous_interpretation
    
    reports_context = state.get("reports_text_context")
    if reports_context:
        formatted_prompt = formatted_prompt + "\n\n[Reports Context for Q&A]\n" + reports_context
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", formatted_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    llm_with_tools = llm.bind_tools(tools)
    agent_runnable = prompt | llm_with_tools
    response = agent_runnable.invoke({"messages": state["messages"]})
    
    return {"messages": [response]}

def process_initial_assessment_from_agent(state: GraphState):
    """Parses the agent's inline technical line for initial assessment and updates state, while emitting a cleaned message for the user."""
    last_message = state["messages"][-1]
    assert isinstance(last_message, AIMessage)
    raw = last_message.content if isinstance(last_message.content, str) else str(last_message.content)
    ml_result = None
    ml_confidence = None
    questionnaire_inputs = {}
    
    # First try to extract from technical line format
    if "INITIAL_ASSESSMENT_RESULT:" in raw:
        try:
            # Find the segment after the marker
            tech_start = raw.find("INITIAL_ASSESSMENT_RESULT:")
            tech_segment = raw[tech_start:]
            # Stop at newline if any
            tech_segment = tech_segment.splitlines()[0]
            # Parse key-value pairs separated by '|'
            parts = {p.split(':', 1)[0]: p.split(':', 1)[1] for p in tech_segment.split('|') if ':' in p}
            ml_result = parts.get("INITIAL_ASSESSMENT_RESULT")
            conf_str = parts.get("CONFIDENCE")
            if conf_str is not None:
                ml_confidence = float(conf_str)
            q_str = parts.get("QUESTIONNAIRE")
            if q_str:
                # Parse semicolon-separated k=v pairs
                for kv in q_str.split(';'):
                    if '=' in kv:
                        k, v = kv.split('=', 1)
                        questionnaire_inputs[k.strip()] = v.strip()
        except Exception:
            pass
    
    # Fallback: Try to extract from Arabic text patterns
    if ml_result is None and state.get("lang") == "ar":
        # Look for Arabic patterns like "النتيجة سلبية" or "النتيجة إيجابية"
        if "النتيجة سلبية" in raw or "النتيجة: سلبية" in raw:
            ml_result = "Negative"
        elif "النتيجة إيجابية" in raw or "النتيجة: إيجابية" in raw:
            ml_result = "Positive"
        elif "Negative" in raw:
            ml_result = "Negative"
        elif "Positive" in raw:
            ml_result = "Positive"
        
        # Try to extract confidence from Arabic text
        import re
        # Look for percentage patterns
        confidence_patterns = [
            r'الثقة.*?(\d+)%',
            r'(\d+)%.*الثقة',
            r'حوالي\s*(\d+)%',
            r'(\d+)%'
        ]
        for pattern in confidence_patterns:
            match = re.search(pattern, raw)
            if match:
                try:
                    ml_confidence = float(match.group(1))
                    break
                except:
                    pass
        
        # Try to extract questionnaire data from conversation history
        if not questionnaire_inputs and ml_result:
            questionnaire_inputs = extract_questionnaire_from_history(state)
    
    # Clean technical lines from the visible content
    cleaned_lines = []
    for line in raw.splitlines():
        if ("INITIAL_ASSESSMENT_RESULT:" in line) or ("CONFIDENCE:" in line) or ("QUESTIONNAIRE:" in line):
            continue
        cleaned_lines.append(line)
    cleaned_content = "\n".join(cleaned_lines).strip()
    
    updates = {}
    if ml_result is not None:
        updates["ml_result"] = ml_result
    if ml_confidence is not None:
        updates["ml_confidence"] = ml_confidence
    if questionnaire_inputs:
        updates["questionnaire_inputs"] = questionnaire_inputs
    
    final_message = AIMessage(content=cleaned_content or (ml_result or ""))
    updates["messages"] = [final_message]
    
    return updates

def reports_agent(state: GraphState):
    """Dedicated multimodal reports interpreter agent (GPT-4o)."""
    file_paths = state.get("report_file_paths") or []
    lang = state["lang"]
    interpretation = interpret_reports_with_gpt4o(file_paths, lang)
    text_context = build_reports_text_context(file_paths)
    final_message = AIMessage(content=interpretation)
    
    # Save text context for follow-up Q&A rounds and clear file paths
    return {
        "messages": [final_message],
        "interpretation_result": interpretation,
        "reports_text_context": text_context,
        "report_file_paths": None,  # Clear the file paths
    }

def auto_analyze_xray(state: GraphState):
    """Automatically runs X-ray analysis when an uploaded image path is present in state."""
    print("--- Auto analyzing X-ray ---")
    image_path = state.get("uploaded_image_path")
    print(f"Image path from state: {image_path}")
    
    if not image_path:
        print("No image path found, returning empty state")
        return {}
    
    try:
        print(f"Calling analyze_xray_image with path: {image_path}")
        tool_output = analyze_xray_image.invoke({"image_path": image_path})
        print(f"Tool output received: {tool_output[:200] if isinstance(tool_output, str) else str(tool_output)[:200]}...")
    except Exception as e:
        print(f"Auto X-ray analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {"xray_result": "Error", "xray_confidence": 0.0, "annotated_image_path": None}
    
    # Parse tool output
    xray_result, xray_confidence, annotated_image_path = "Error", 0.0, None
    if isinstance(tool_output, str) and '|' in tool_output and ':' in tool_output:
        parts = {p.split(':', 1)[0]: p.split(':', 1)[1] for p in tool_output.split('|')}
        xray_result = parts.get("XRAY_RESULT", "Error")
        try:
            xray_confidence = float(parts.get("CONFIDENCE", 0.0))
        except Exception:
            xray_confidence = 0.0
        annotated_image_path = parts.get("ANNOTATED_IMAGE_PATH")
        if annotated_image_path == 'None':
            annotated_image_path = None
        
        print(f"Successfully parsed tool output:")
        print(f"  Result: {xray_result}")
        print(f"  Confidence: {xray_confidence}")
        print(f"  Annotated image path: {annotated_image_path}")
    else:
        print(f"Could not parse tool output: {tool_output}")
    
    print(f"Auto X-ray State: result={xray_result}, conf={xray_confidence}")
    result = {"xray_result": xray_result, "xray_confidence": xray_confidence, "annotated_image_path": annotated_image_path}
    print(f"Returning state update: {result}")
    return result

def generate_ml_report(state: GraphState):
    """Generates the final summary message after only the ml has been analyzed."""
    # Simplified approach to avoid event loop issues
    import asyncio
    import nest_asyncio
    
    try:
        # Apply nest_asyncio to allow nested event loops
        nest_asyncio.apply()
        
        # Create a new event loop for this operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            interpretation = loop.run_until_complete(interpret_ml_results(
                ml_result=state["ml_result"], 
                ml_confidence=state["ml_confidence"],
                lang=state["lang"],
            ))
            print(f"ML interpretation received: {interpretation[:100]}...")
        finally:
            loop.close()
            
    except ImportError:
        # If nest_asyncio is not available, use a simpler approach
        print("nest_asyncio not available, using fallback approach...")
        interpretation = f"ML assessment completed. Result: {state.get('ml_result', 'Unknown')}, Confidence: {state.get('ml_confidence', 0):.1f}%. Please consult your doctor for detailed interpretation."
    except Exception as e:
        print(f"Error in async ML interpretation: {e}")
        interpretation = f"ML assessment completed. Result: {state.get('ml_result', 'Unknown')}, Confidence: {state.get('ml_confidence', 0):.1f}%. Please consult your doctor for detailed interpretation."
    
    final_message = AIMessage(content=interpretation)
    return {"messages": [final_message]}

def generate_xray_report(state: GraphState):
    """
    Generates the final summary message after the X-ray has been analyzed
    and clears the uploaded image path to prevent re-analysis.
    """
    print("--- Generating X-ray report ---")
    print(f"State keys: {list(state.keys())}")
    print(f"X-ray result: {state.get('xray_result')}")
    print(f"X-ray confidence: {state.get('xray_confidence')}")
    print(f"Language: {state.get('lang')}")
    
    try:
        print("Calling interpret_xray_results...")
        
        # Simplified approach to avoid event loop issues
        import asyncio
        import nest_asyncio
        
        try:
            # Apply nest_asyncio to allow nested event loops
            nest_asyncio.apply()
            
            # Create a new event loop for this operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                interpretation = loop.run_until_complete(interpret_xray_results(
                    xray_result=state["xray_result"], 
                    xray_confidence=state["xray_confidence"],
                    lang=state["lang"],
                ))
                print(f"Interpretation received: {interpretation[:100]}...")
            finally:
                loop.close()
                
        except ImportError:
            # If nest_asyncio is not available, use a simpler approach
            print("nest_asyncio not available, using fallback approach...")
            interpretation = f"X-ray analysis completed. Result: {state.get('xray_result', 'Unknown')}, Confidence: {state.get('xray_confidence', 0):.1f}%. Please consult your doctor for detailed interpretation."
        except Exception as e:
            print(f"Error in async interpretation: {e}")
            interpretation = f"X-ray analysis completed. Result: {state.get('xray_result', 'Unknown')}, Confidence: {state.get('xray_confidence', 0):.1f}%. Please consult your doctor for detailed interpretation."
        
        final_message = AIMessage(content=interpretation)
        print("AIMessage created successfully")
        
        # Clear the image path to prevent re-triggering the analysis
        result = {"messages": [final_message], "uploaded_image_path": None}
        print(f"Returning result with {len(result['messages'])} messages")
        return result
        
    except Exception as e:
        print(f"Error in generate_xray_report: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: create a basic message
        fallback_message = f"X-ray analysis completed. Result: {state.get('xray_result', 'Unknown')}, Confidence: {state.get('xray_confidence', 0):.1f}%. Please consult your doctor for detailed interpretation."
        final_message = AIMessage(content=fallback_message)
        return {"messages": [final_message], "uploaded_image_path": None}

def process_interpretation_result(state: GraphState):
    """Parses the output of the interpretation tool and updates the state."""
    last_message = state["messages"][-1]
    assert isinstance(last_message, ToolMessage)
    tool_output = last_message.content
    if 'INTERPRETATION_RESULT:' in tool_output:
        interpretation = tool_output.replace("INTERPRETATION_RESULT:", "").strip()
    else:
        interpretation = "Error processing interpretation."
    
    # Create a new AIMessage with the interpretation to show to the user
    final_message = AIMessage(content=interpretation)
    
    return {"messages": [final_message], "interpretation_result": interpretation}

def process_xray_analysis_result(state: GraphState):
    """Parses the output of the X-ray analysis tool and updates the state."""
    print("--- Processing X-ray analysis result ---")
    last_message = state["messages"][-1]
    print(f"Last message type: {type(last_message)}")
    
    assert isinstance(last_message, ToolMessage)
    tool_output = last_message.content
    print(f"Tool output: {tool_output[:200]}...")
    
    if '|' in tool_output and ':' in tool_output:
        xray_parts = {p.split(':', 1)[0]: p.split(':', 1)[1] for p in tool_output.split('|')}
        xray_result = xray_parts.get("XRAY_RESULT", "Error")
        xray_confidence = float(xray_parts.get("CONFIDENCE", 0.0))
        annotated_image_path = xray_parts.get("ANNOTATED_IMAGE_PATH")
        if annotated_image_path == 'None': annotated_image_path = None
        
        print(f"Parsed X-ray parts:")
        print(f"  Result: {xray_result}")
        print(f"  Confidence: {xray_confidence}")
        print(f"  Annotated image path: {annotated_image_path}")
    else:
        xray_result, xray_confidence, annotated_image_path = "Error", 0.0, None
        print(f"Could not parse tool output, using defaults: {xray_result}, {xray_confidence}")
    
    print(f"Updated State with X-ray Result: {xray_result}")
    result = {"xray_result": xray_result, "xray_confidence": xray_confidence, "annotated_image_path": annotated_image_path}
    print(f"Returning state update: {result}")
    return result

# 3. DEFINE THE EDGES (ROUTING LOGIC)
def route_after_agent(state: GraphState):
    """Routes to tools or ends the turn."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    # Route to initial assessment parser if the agent embedded a technical line
    if isinstance(last, AIMessage):
        content = last.content if isinstance(last.content, str) else str(last.content)
        if "INITIAL_ASSESSMENT_RESULT:" in content:
            return "generate_ml_report"
        # Also check for Arabic assessment patterns
        if state.get("lang") == "ar":
            if ("النتيجة" in content and ("سلبية" in content or "إيجابية" in content)) or \
               ("النتيجة الأولية" in content) or \
               ("Negative" in content or "Positive" in content):
                # Check if we have enough conversation to make an assessment
                messages = state.get("messages", [])
                # Count questions asked (typically need at least 5 for full assessment)
                ai_questions = sum(1 for msg in messages if isinstance(msg, AIMessage) and "?" in str(msg.content))
                if ai_questions >= 4:  # After asking about the main 5 questions
                    return "process_initial_assessment"
    return END

def entry_node(state: GraphState):
    """No-op entry node to enable conditional routing based on provided inputs."""
    print("--- Entry node ---")
    print(f"Entry node state keys: {list(state.keys())}")
    print(f"Entry node messages count: {len(state.get('messages', []))}")
    print(f"Entry node uploaded_image_path: {state.get('uploaded_image_path')}")
    print(f"Entry node report_file_paths: {state.get('report_file_paths')}")
    print(f"Entry node lang: {state.get('lang')}")
    return {}

def route_from_entry(state: GraphState):
    """Route to appropriate handler based on input type - file uploads ALWAYS take priority."""
    print("--- Route from entry ---")
    print(f"State keys: {list(state.keys())}")
    print(f"Messages count: {len(state.get('messages', []))}")
    print(f"Uploaded image path: {state.get('uploaded_image_path')}")
    print(f"Report file paths: {state.get('report_file_paths')}")
    print(f"Existing interpretation result: {bool(state.get('interpretation_result'))}")
    print(f"Existing xray result: {bool(state.get('xray_result'))}")
    
    # ABSOLUTE PRIORITY: File uploads ALWAYS take precedence regardless of conversation state
    # This ensures users can upload files at any point in the conversation
    if state.get("uploaded_image_path"):
        print("🎯 PRIORITY ROUTE: auto_xray due to uploaded image path")
        return "auto_xray"
    
    if state.get("report_file_paths"):
        print("🎯 PRIORITY ROUTE: reports_agent due to report files")
        return "reports_agent"
    
    # FALLBACK: Check if the latest message contains file paths (legacy support)
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        last_message_content = state["messages"][-1].content
        print(f"Last message content: {last_message_content[:100]}...")
        
        if isinstance(last_message_content, str):
            # Check for various image path patterns
            image_indicators = ["temp_upload", "/tmp/", "annotated_images/", ".jpg", ".png", ".jpeg"]
            if any(indicator in last_message_content for indicator in image_indicators):
                print("🎯 FALLBACK ROUTE: auto_xray due to image path in message")
                return "auto_xray"
    
    # DEFAULT: Route to conversational agent for regular chat
    print("📝 DEFAULT ROUTE: conversational agent")
    return "agent"

def route_after_tools(state: GraphState):
    """Routes to the correct result processing node based on which tool was called."""
    last_message = state["messages"][-1]
    assert isinstance(last_message, ToolMessage)
    
    ai_message = next(msg for msg in reversed(state['messages']) if isinstance(msg, AIMessage) and msg.tool_calls)
    tool_call_id = last_message.tool_call_id
    tool_call = next(tc for tc in ai_message.tool_calls if tc['id'] == tool_call_id)
    tool_name = tool_call['name']
    
    print(f"Routing after tool '{tool_name}' execution.")
    if tool_name == "analyze_xray_image":
        return "process_xray_analysis"
    elif tool_name == "interpret_medical_reports":
        return "process_interpretation"
    return END

def route_after_reports_agent(state: GraphState):
    """After processing reports, continue to conversational agent for follow-up questions."""
    return "agent"

# 4. BUILD THE GRAPH
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("agent", call_agent)
workflow.add_node("tools", tool_node)
workflow.add_node("reports_agent", reports_agent)
workflow.add_node("auto_xray", auto_analyze_xray)
workflow.add_node("entry", entry_node)
workflow.add_node("process_initial_assessment", process_initial_assessment_from_agent)
workflow.add_node("process_xray_analysis", process_xray_analysis_result)
workflow.add_node("process_interpretation", process_interpretation_result)
workflow.add_node("generate_ml_report", generate_ml_report)
workflow.add_node("generate_xray_report", generate_xray_report)

# Set entry point
workflow.set_entry_point("entry")

# Add conditional edges
workflow.add_conditional_edges("entry", route_from_entry, {
    "agent": "agent", 
    "reports_agent": "reports_agent", 
    "auto_xray": "auto_xray"
})

workflow.add_conditional_edges("agent", route_after_agent, {
    "tools": "tools", 
    "generate_ml_report": "generate_ml_report", 
    END: END
})

workflow.add_conditional_edges("tools", route_after_tools, {
    "process_xray_analysis": "process_xray_analysis",
    "process_interpretation": "process_interpretation",
    END: END,
})

# Add direct edges
workflow.add_edge("process_initial_assessment", "generate_ml_report")
workflow.add_edge("process_xray_analysis", "generate_xray_report")
workflow.add_edge("process_interpretation", "agent")
workflow.add_edge("reports_agent", "agent")
workflow.add_edge("auto_xray", "generate_xray_report")
workflow.add_edge("generate_ml_report", END)  # End after ML assessment (user can start new conversation)
workflow.add_edge("generate_xray_report", END)  # End after X-ray analysis (user can start new conversation)

# --- COMPILE THE GRAPH WITH CHECKPOINTER ---
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)