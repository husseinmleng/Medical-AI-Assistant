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
llm = ChatOpenAI(model="gpt-4.1", temperature=0.4, max_tokens=2000)
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
   - At the end, add a single separate technical line (for internal use only):
     INITIAL_ASSESSMENT_RESULT:<Positive|Negative>|CONFIDENCE:<0-100>|QUESTIONNAIRE:family_history_breast_cancer=<yes/no>;recent_weight_loss=<yes/no>;previous_breast_conditions=<yes/no>;symptom_duration_days=<number>;fatigue=<yes/no>
   - Do NOT mention this technical line in your visible message.
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
"""
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
    ml_result = state.get('ml_result', 'Not yet available') # Get ml_result from state
    has_reports = "Yes" if state.get("interpretation_result") else "No"
    formatted_prompt = system_prompt_template.format(
        language=language_map.get(lang, "English"),
        ml_result=ml_result,
        has_reports=has_reports
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
    # Extract technical line components
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
    image_path = state.get("uploaded_image_path")
    if not image_path:
        return {}
    try:
        tool_output = analyze_xray_image.invoke({"image_path": image_path})
    except Exception as e:
        print(f"Auto X-ray analysis failed: {e}")
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
    print(f"Auto X-ray State: result={xray_result}, conf={xray_confidence}")
    return {"xray_result": xray_result, "xray_confidence": xray_confidence, "annotated_image_path": annotated_image_path}
def generate_ml_report(state: GraphState):
    """Generates the final summary message after only the ml has been analyzed."""
    interpretation = asyncio.run(interpret_ml_results(
        ml_result=state["ml_result"], ml_confidence=state["ml_confidence"],
        lang=state["lang"],
    ))
    final_message = AIMessage(content=interpretation)
    return {"messages": [final_message]}

def generate_xray_report(state: GraphState):
    """
    Generates the final summary message after the X-ray has been analyzed
    and clears the uploaded image path to prevent re-analysis.
    """
    interpretation = asyncio.run(interpret_xray_results(
        xray_result=state["xray_result"], xray_confidence=state["xray_confidence"],
        lang=state["lang"],
    ))
    final_message = AIMessage(content=interpretation)
    # --- FIX: Clear the image path to prevent re-triggering the analysis ---
    return {"messages": [final_message], "uploaded_image_path": None}

def process_risk_prediction_result(state: GraphState):
    """Parses the output of the first tool and updates the state."""
    last_message = state["messages"][-1]
    assert isinstance(last_message, ToolMessage)
    tool_output = last_message.content
    ai_message_with_tool_call = next(msg for msg in reversed(state["messages"]) if isinstance(msg, AIMessage) and msg.tool_calls)
    questionnaire_inputs = ai_message_with_tool_call.tool_calls[0]['args']
    if '|' in tool_output and ':' in tool_output:
        parts = {p.split(':', 1)[0]: p.split(':', 1)[1] for p in tool_output.split('|')}
        ml_result = parts.get("INITIAL_ASSESSMENT_RESULT", "Error")
        ml_confidence = float(parts.get("CONFIDENCE", 0.0))
    else:
        ml_result, ml_confidence = "Error", 0.0
    print(f"Updated State with ML Result: {ml_result}")
    return {"questionnaire_inputs": questionnaire_inputs, "ml_result": ml_result, "ml_confidence": ml_confidence}
def process_xray_analysis_result(state: GraphState):
    """Parses the output of the second tool and updates the state."""
    last_message = state["messages"][-1]
    assert isinstance(last_message, ToolMessage)
    tool_output = last_message.content
    if '|' in tool_output and ':' in tool_output:
        xray_parts = {p.split(':', 1)[0]: p.split(':', 1)[1] for p in tool_output.split('|')}
        xray_result = xray_parts.get("XRAY_RESULT", "Error")
        xray_confidence = float(xray_parts.get("CONFIDENCE", 0.0))
        annotated_image_path = xray_parts.get("ANNOTATED_IMAGE_PATH")
        if annotated_image_path == 'None': annotated_image_path = None
    else:
        xray_result, xray_confidence, annotated_image_path = "Error", 0.0, None
    print(f"Updated State with X-ray Result: {xray_result}")
    return {"xray_result": xray_result, "xray_confidence": xray_confidence, "annotated_image_path": annotated_image_path}
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
    return END
def entry_node(state: GraphState):
    """No-op entry node to enable conditional routing based on provided inputs."""
    return {}
def route_from_entry(state: GraphState):
    """Route to reports agent immediately when NEW report files are present, otherwise to the conversational agent."""
    # Only route to reports agent if NEW files are uploaded and there's no prior interpretation
    if state.get("report_file_paths") and not state.get("interpretation_result"):
        return "reports_agent"
    if state.get("uploaded_image_path"):
        return "auto_xray"
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
workflow.set_entry_point("entry")
workflow.add_conditional_edges("entry", route_from_entry, {"agent": "agent", "reports_agent": "reports_agent", "auto_xray": "auto_xray"})
workflow.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "generate_ml_report": "generate_ml_report", END: END})
workflow.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "process_xray_analysis": "process_xray_analysis",
        "process_interpretation": "process_interpretation",
        END: END,
    }
)
workflow.add_edge("process_initial_assessment", "generate_ml_report")
workflow.add_edge("process_xray_analysis", "generate_xray_report")
workflow.add_edge("process_interpretation", "agent")
workflow.add_edge("reports_agent", "agent")
workflow.add_edge("auto_xray", "generate_xray_report")
workflow.add_edge("generate_ml_report", END)
workflow.add_edge("generate_xray_report", END)
# --- COMPILE THE GRAPH WITH CHECKPOINTER ---
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)