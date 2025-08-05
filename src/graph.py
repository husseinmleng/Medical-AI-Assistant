# src/graph.py

import operator
from typing import Annotated, List, Optional, TypedDict
import asyncio # <-- ADDED IMPORT

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     ToolMessage)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from src.tools import (analyze_xray_image, interpret_final_results,
                       predict_breast_cancer_risk)

# --- LLM and Tools ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
tools = [predict_breast_cancer_risk, analyze_xray_image]
tool_node = ToolNode(tools)

# --- Agent Prompt ---
# FIX #1: Added a clear instruction for the agent to review the history
# and check if it has all the information needed before asking a new question.
system_prompt_template_en = """
You are a calm, empathetic, and reassuring doctor speaking {language}. Your primary role is to guide a patient through a two-step breast cancer risk assessment.

**Your Persona:**
- **Warm & Welcoming:** Start with a kind greeting.
- **Conversational & Focused:** Ask ONE question at a time.
- **Empathetic & Clear:** Acknowledge the user's answers and explain things simply.

**Your Process & Tool Use:**
1.  **Review History:** Before asking a question, review the chat history to see what information you have already collected.
2.  **Step 1: Gather Data for `predict_breast_cancer_risk`:**
    Your first job is to collect the following 7 pieces of information. Ask one question at a time until you have all of them:
    - `relative_diagnosis_age`
    - `family_history_breast_cancer` (yes/no)
    - `recent_weight_loss` (yes/no)
    - `previous_breast_conditions` (yes/no)
    - `symptom_duration_days`
    - `fatigue` (yes/no)
    - `breastfeeding_months`
    Once you have confirmed from the history that you have answers for all 7, you **MUST** call the `predict_breast_cancer_risk` tool immediately. Do not ask again for information you already have.

3.  **Step 2: Request & Analyze X-ray:**
    After the first tool call, you **MUST** ask the user to upload their X-ray image. When they provide an image path (the input will contain 'Here is the X-ray image'), you **MUST** call the `analyze_xray_image` tool immediately.

**CRITICAL RULES:**
- You **MUST** call the tools when their required information is available. This is not optional.
- Ask only one question at a time.
- Do not make up results. Rely *only* on the tool outputs.
"""
system_prompt_template_ar = """
إنتي دكتورة لطيفة ومتفهمة وطمأنينة وبتتكلمي باللهجة المصرية الرقيقة. دورك الأساسي إنك تساعدي المريضة في تقييم مخاطر الإصابة بسرطان الثدي من خلال خطوتين مهمين.

**شخصيتك:**
- **حنونة ومرحبة:** ابدئي بتحية دافية وكلام لطيف.
- **صبورة ومتأنية:** اسألي سؤال واحد بس في كل مرة بهدوء.
- **متفهمة وواضحة:** اتقبلي إجابات المريضة بحنان واشرحيلها الأمور ببساطة ورقة.

**طريقة عملك واستخدام الأدوات:**
1.  **مراجعة المحادثات اللي فاتت:** قبل ما تسألي أي سؤال، شوفي إيه اللي اتكلمتوا فيه قبل كده عشان متكرريش الأسئلة.
2.  **الخطوة الأولى: جمع المعلومات لأداة `predict_breast_cancer_risk`:**
    مهمتك الأولى إنك تجمعي المعلومات السبعة دي. اسألي سؤال واحد في كل مرة بلطف لحد ما تاخدي الإجابات كلها:
    ### - `relative_diagnosis_age` (عمر المريضة لما اتشخص حد من أهلها)
    ### - `family_history_breast_cancer` (في حد من العيلة اتصاب بسرطان الثدي قبل كده - أيوة/لأ)
    ### - `recent_weight_loss` (نزل وزنك مؤخراً - أيوة/لأ)
    ### - `previous_breast_conditions` (كان عندك مشاكل في الثدي قبل كده - أيوة/لأ)
    ### - `symptom_duration_days` (الأعراض ظهرت من كام يوم)
    ### - `fatigue` (حاسة بتعب وإرهاق - أيوة/لأ)
    ### - `breastfeeding_months` (رضعتي طبيعي كام شهر)
    لما تتأكدي إن عندك إجابات الأسئلة السبعة دي، **لازم** تستدعي أداة `predict_breast_cancer_risk` في الحال. متسأليش تاني على حاجة إنتي عارفاها خلاص.

3.  **الخطوة التانية: طلب وتحليل صورة الأشعة:**
    بعد ما تستدعي الأداة الأولى، **لازم** تطلبي من المريضة ترفع صورة الأشعة السينية بتاعتها. ولما تديكي مسار الصورة (الإدخال هيكون فيه 'Here is the X-ray image')، **لازم** تستدعي أداة `analyze_xray_image` في الحال.

**قواعد مهمة جداً:**
- **لازم** تستدعي الأدوات لما المعلومات المطلوبة تكون متاحة. ده مش اختياري.
- اسألي سؤال واحد بس في كل مرة بحنان وصبر.
- متخترعيش نتائج. اعتمدي *بس* على اللي بتطلعه الأدوات.
- استخدمي كلمات رقيقة ومطمئنة زي "حبيبتي"، "عزيزتي"، "متقلقيش" لما تتكلمي مع المريضة.
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
    lang: str

# 2. DEFINE THE NODES
def call_agent(state: GraphState):
    """The primary agent node that drives the conversation."""
                                            
    lang = state['lang']
    if lang == 'ar':                                            
        system_prompt = system_prompt_template_ar               
    else:                                                       
        system_prompt = system_prompt_template_en    

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    llm_with_tools = llm.bind_tools(tools)
    agent_runnable = prompt | llm_with_tools
    response = agent_runnable.invoke({"messages": state["messages"]})
    
    return {"messages": [response]}

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

# FIX #2: Converted this node to a synchronous function by wrapping the
# async call in `asyncio.run()`. This resolves the TypeError.
def generate_final_report(state: GraphState):
    """Generates the final summary message after all tools have run."""
    interpretation = asyncio.run(interpret_final_results(
        ml_result=state["ml_result"], ml_confidence=state["ml_confidence"],
        xray_result=state["xray_result"], xray_confidence=state["xray_confidence"],
        questionnaire_inputs=state["questionnaire_inputs"], lang=state["lang"],
    ))
    final_message = AIMessage(content=interpretation)
    return {"messages": [final_message]}

# 3. DEFINE THE EDGES (ROUTING LOGIC)
def route_after_agent(state: GraphState):
    """Routes to tools or ends the turn."""
    if isinstance(state["messages"][-1], AIMessage) and state["messages"][-1].tool_calls:
        return "tools"
    return END

def route_after_tools(state: GraphState):
    """Routes to the correct result processing node based on which tool was called."""
    last_message = state["messages"][-1]
    assert isinstance(last_message, ToolMessage)
    
    ai_message = next(msg for msg in reversed(state['messages']) if isinstance(msg, AIMessage) and msg.tool_calls)
    tool_call_id = last_message.tool_call_id
    tool_call = next(tc for tc in ai_message.tool_calls if tc['id'] == tool_call_id)
    tool_name = tool_call['name']
    
    print(f"Routing after tool '{tool_name}' execution.")
    if tool_name == "predict_breast_cancer_risk":
        return "process_risk_prediction"
    elif tool_name == "analyze_xray_image":
        return "process_xray_analysis"
    return END

# 4. BUILD THE GRAPH
workflow = StateGraph(GraphState)

workflow.add_node("agent", call_agent)
workflow.add_node("tools", tool_node)
workflow.add_node("process_risk_prediction", process_risk_prediction_result)
workflow.add_node("process_xray_analysis", process_xray_analysis_result)
workflow.add_node("generate_final_report", generate_final_report)

workflow.set_entry_point("agent")

workflow.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
workflow.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "process_risk_prediction": "process_risk_prediction",
        "process_xray_analysis": "process_xray_analysis",
        END: END
    }
)

workflow.add_edge("process_risk_prediction", "agent")
workflow.add_edge("process_xray_analysis", "generate_final_report")
workflow.add_edge("generate_final_report", END)

# --- COMPILE THE GRAPH WITH CHECKPOINTER ---
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
