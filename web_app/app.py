from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import asyncio
from typing import AsyncGenerator, Annotated
from contextlib import asynccontextmanager
from inference.exec_graph import ChatLangGraph
from langsmith import tracing_context
from dotenv import load_dotenv

load_dotenv()

# Initialize the chatbot
chatbot: ChatLangGraph | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global chatbot
    with tracing_context(enabled=True, project_name="langchain-pepwave"):
        chatbot = ChatLangGraph(
            llm_model="gpt-4o-mini",
            pinecone_index_name="pepwave-early-april-page-content-embedding",
        )
    yield
    # Shutdown (cleanup if needed)
    chatbot = None


app = FastAPI(
    title="Pepwave ChatBot",
    description="RAG-powered chatbot with LangGraph",
    lifespan=lifespan,
)

# Use relative paths within the web_app directory
templates = Jinja2Templates(directory="web_app/templates")
app.mount("/static", StaticFiles(directory="web_app/static"), name="static")


# Dependency to get chatbot instance
def get_chatbot() -> ChatLangGraph:
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    return chatbot


class ChatMessage(BaseModel):
    message: str
    thread_id: str | None = None


class ThreadCreate(BaseModel):
    title: str | None = None


class ThreadResponse(BaseModel):
    thread_id: str
    title: str


class ThreadInfo(BaseModel):
    thread_id: str
    title: str
    message_count: int
    created_at: str


class ThreadsResponse(BaseModel):
    threads: list[ThreadInfo]


class MessageInfo(BaseModel):
    type: str
    content: str
    timestamp: str | None = None


class ThreadHistoryResponse(BaseModel):
    messages: list[MessageInfo]


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/threads", response_model=ThreadResponse, status_code=201)
async def create_thread(
    thread_data: ThreadCreate, bot: Annotated[ChatLangGraph, Depends(get_chatbot)]
):
    """Create a new conversation thread."""
    thread_id = bot.create_new_thread()
    if thread_data.title:
        bot.active_threads[thread_id]["title"] = thread_data.title
    return ThreadResponse(
        thread_id=thread_id, title=bot.active_threads[thread_id]["title"]
    )


@app.get("/api/threads", response_model=ThreadsResponse)
async def list_threads(bot: Annotated[ChatLangGraph, Depends(get_chatbot)]):
    """List all conversation threads."""
    threads = []
    for thread_id, info in bot.list_threads().items():
        threads.append(
            ThreadInfo(
                thread_id=thread_id,
                title=info["title"],
                message_count=bot.get_thread_message_count(thread_id),
                created_at=info["created_at"].isoformat(),
            )
        )
    return ThreadsResponse(threads=threads)


@app.get("/api/threads/{thread_id}/history", response_model=ThreadHistoryResponse)
async def get_thread_history(
    thread_id: str, bot: Annotated[ChatLangGraph, Depends(get_chatbot)]
):
    """Get conversation history for a specific thread."""
    history = bot.get_thread_history(thread_id)
    messages = []
    for msg in history:
        messages.append(
            MessageInfo(
                type=msg.type,
                content=msg.content,
                timestamp=getattr(msg, 'timestamp', None),
            )
        )
    return ThreadHistoryResponse(messages=messages)


@app.post("/api/chat/stream")
async def chat_stream(
    chat_message: ChatMessage, bot: Annotated[ChatLangGraph, Depends(get_chatbot)]
):
    """Stream chat response using Server-Sent Events."""

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
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
