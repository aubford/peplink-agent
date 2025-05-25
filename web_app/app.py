from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import asyncio
from typing import AsyncGenerator
from inference.exec_graph import ChatLangGraph
from langsmith import tracing_context
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Pepwave ChatBot", description="RAG-powered chatbot with LangGraph")

# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the chatbot
chatbot: ChatLangGraph | None = None


class ChatMessage(BaseModel):
    message: str
    thread_id: str | None = None


class ThreadCreate(BaseModel):
    title: str | None = None


@app.on_event("startup")
async def startup_event():
    global chatbot
    with tracing_context(enabled=True, project_name="langchain-pepwave"):
        chatbot = ChatLangGraph(
            llm_model="gpt-4o-mini",
            pinecone_index_name="pepwave-early-april-page-content-embedding",
        )


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/threads")
async def create_thread(thread_data: ThreadCreate):
    """Create a new conversation thread."""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    thread_id = chatbot.create_new_thread()
    if thread_data.title:
        chatbot.active_threads[thread_id]["title"] = thread_data.title
    return {"thread_id": thread_id, "title": chatbot.active_threads[thread_id]["title"]}


@app.get("/api/threads")
async def list_threads():
    """List all conversation threads."""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    threads = []
    for thread_id, info in chatbot.list_threads().items():
        threads.append(
            {
                "thread_id": thread_id,
                "title": info["title"],
                "message_count": chatbot.get_thread_message_count(thread_id),
                "created_at": info["created_at"].isoformat(),
            }
        )
    return {"threads": threads}


@app.get("/api/threads/{thread_id}/history")
async def get_thread_history(thread_id: str):
    """Get conversation history for a specific thread."""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    history = chatbot.get_thread_history(thread_id)
    messages = []
    for msg in history:
        messages.append(
            {
                "type": msg.type,
                "content": msg.content,
                "timestamp": getattr(msg, 'timestamp', None),
            }
        )
    return {"messages": messages}


@app.post("/api/chat/stream")
async def chat_stream(chat_message: ChatMessage):
    """Stream chat response using Server-Sent Events."""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    # Capture chatbot reference for use in nested function
    bot = chatbot

    async def generate_response() -> AsyncGenerator[str, None]:
        try:
            # Send initial event to indicate streaming started
            yield f"data: {json.dumps({'type': 'start'})}\n\n"

            # Collect the full response for potential use
            full_response = ""

            # Stream the response from the chatbot
            for token in bot.query(chat_message.message, chat_message.thread_id):
                full_response += token
                # Send each token as a streaming event
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete', 'full_response': full_response})}\n\n"

        except Exception as e:
            # Send error event
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )
