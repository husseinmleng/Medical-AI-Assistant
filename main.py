
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import uuid
import asyncio
from src.app_logic import run_graph, clear_session_history, generate_chat_title, transcribe_audio, get_history

app = FastAPI()

# --- Static Files and Templates ---
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- In-memory storage for conversations ---
conversations = {}

# --- Helper Functions ---
def get_or_create_session(session_id: str = None):
    if session_id and session_id in conversations:
        return session_id, conversations[session_id]
    
    session_id = str(uuid.uuid4())
    conversations[session_id] = {
        "title": "New Chat",
        "lang": None,
        "messages": [],
        "annotated_image_path": None,
    }
    clear_session_history(session_id)
    return session_id, conversations[session_id]

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def handle_chat(request: Request):
    data = await request.json()
    user_input = data.get("message")
    session_id = data.get("session_id")
    
    session_id, active_conv = get_or_create_session(session_id)

    if active_conv["lang"] is None:
        # Simple language detection, you might want a more robust way
        if any(char in "ابجدهوزحطيكلمنسعفصقرشتثخذضظغ" for char in user_input):
             active_conv["lang"] = "ar"
        else:
            active_conv["lang"] = "en"


    # Add user message to the conversation state
    active_conv["messages"].append({"role": "user", "content": user_input})

    # Generate a title for new chats on the first user message
    is_new_chat = len(active_conv["messages"]) < 2
    if is_new_chat:
        try:
            active_conv['title'] = await generate_chat_title(user_input)
        except Exception as e:
            print(f"Error generating title: {e}")
            active_conv['title'] = "Medical Chat"

    # Run the graph with the actual user input
    response = run_graph(user_input, session_id, active_conv["lang"])
    
    final_messages = response.get("messages", [])
    active_conv["messages"] = [{"role": msg.type, "content": msg.content} for msg in final_messages]
    
    return JSONResponse({
        "session_id": session_id,
        "messages": active_conv["messages"],
        "title": active_conv["title"]
    })

@app.post("/upload_image")
async def upload_image(session_id: str, file: UploadFile = File(...)):
    session_id, active_conv = get_or_create_session(session_id)

    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    image_path = os.path.join(temp_dir, file.filename)

    with open(image_path, "wb") as buffer:
        buffer.write(await file.read())

    # Run the graph with the image
    response = run_graph(
        user_input="Here is the X-ray image for analysis.",
        session_id=session_id,
        lang=active_conv.get("lang", "en"),
        image_path=image_path
    )

    final_messages = response.get("messages", [])
    active_conv["messages"] = [{"role": msg.type, "content": msg.content} for msg in final_messages]
    active_conv["annotated_image_path"] = response.get("annotated_image_path")
    
    # Clean up the temporary file
    if os.path.exists(image_path):
        os.remove(image_path)

    return JSONResponse({
        "session_id": session_id,
        "messages": active_conv["messages"],
        "annotated_image_path": active_conv["annotated_image_path"]
    })

@app.get("/history/{session_id}")
async def get_chat_history(session_id: str):
    if session_id in conversations:
        return JSONResponse(conversations[session_id])
    return JSONResponse({"error": "Session not found"}, status_code=404)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
