import sys
import os

from langgraph.checkpoint.memory import InMemorySaver

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import asyncio
import random
from typing import AsyncGenerator, Annotated
from contextlib import asynccontextmanager
from inference.chat_agentic import ChatLangGraph
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# from langsmith import tracing_context

load_dotenv()

# Initialize the chatbot
chatbot: ChatLangGraph | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global chatbot

    if os.getenv("USE_IN_MEMORY_CHECKPOINTER") == "true":
        print("✅ Using in-memory checkpointer")
        chatbot = ChatLangGraph(
            llm_model="gpt-5.2",
            pinecone_index_name="pepwave-early-april-page-content-embedding",
            checkpointer=InMemorySaver(),
        )
        yield

    else:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is required")

        async with AsyncPostgresSaver.from_conn_string(database_url) as checkpointer:
            await checkpointer.setup()
            print("✅ Using PostgreSQL for persistence")

            chatbot = ChatLangGraph(
                llm_model="gpt-5.2",
                pinecone_index_name="pepwave-early-april-page-content-embedding",
                checkpointer=checkpointer,
            )
            print("✅ Chatbot initialized")
            yield

    # Shutdown (cleanup if needed)
    chatbot = None


app = FastAPI(
    title="Pepwave ChatBot",
    description="RAG-powered chatbot with LangGraph",
    lifespan=lifespan,
)

# Use absolute paths based on the current file location
web_app_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(web_app_dir, "templates"))
app.mount(
    "/static", StaticFiles(directory=os.path.join(web_app_dir, "static")), name="static"
)


# Dependency to get chatbot instance
def get_chatbot() -> ChatLangGraph:
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    return chatbot


class ChatMessage(BaseModel):
    message: str
    thread_id: str


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


class TestsetQuery(BaseModel):
    query: str
    answer: str


class TestsetQueriesResponse(BaseModel):
    queries: list[TestsetQuery]


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/testset-queries", response_model=TestsetQueriesResponse)
async def get_random_testset_queries():
    """Get random queries from the testset for suggestions."""
    try:
        # Fix the path - we need to go up one directory from web_app to langchain-pepwave root
        testset_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "evals/testsets/testset-200_main_testset_25-04-23/generated_testset.json",
        )

        print(f"Looking for testset at: {testset_path}")  # Debug logging

        if not os.path.exists(testset_path):
            print(f"Testset file not found at: {testset_path}")  # Debug logging
            raise HTTPException(
                status_code=404, detail=f"Testset file not found at: {testset_path}"
            )

        with open(testset_path, 'r', encoding='utf-8') as f:
            testset_data = json.load(f)

        # Extract all queries
        all_queries = []
        for item in testset_data:
            all_queries.append(TestsetQuery(query=item['query'], answer=item['answer']))

        # Return 6 random queries for suggestions
        if len(all_queries) >= 6:
            random_queries = random.sample(all_queries, 6)
        else:
            random_queries = all_queries

        return TestsetQueriesResponse(queries=random_queries)

    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except Exception as e:
        # Log the actual error and include it in the response
        print(f"Error loading testset queries: {str(e)}")  # Debug logging
        raise HTTPException(
            status_code=500, detail=f"Failed to load testset queries: {str(e)}"
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


@app.delete("/api/threads/{thread_id}", status_code=204)
async def delete_thread(
    thread_id: str, bot: Annotated[ChatLangGraph, Depends(get_chatbot)]
):
    """Delete a conversation thread."""
    try:
        bot.delete_thread(thread_id)
        return None  # 204 No Content
    except KeyError:
        raise HTTPException(status_code=404, detail="Thread not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete thread: {str(e)}"
        )


@app.post("/api/chat/stream")
async def chat_stream(
    chat_message: ChatMessage, graph: Annotated[ChatLangGraph, Depends(get_chatbot)]
):
    """Stream chat response using Server-Sent Events."""

    async def generate_response() -> AsyncGenerator[str, None]:
        try:
            # Send initial event to indicate streaming started
            yield f"data: {json.dumps({'type': 'start'})}\n\n"

            # Stream the response from the chatbot
            async for chunk in graph.query(
                chat_message.message, chat_message.thread_id
            ):
                yield f"data: {chunk}\n\n"
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.001)

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            # Send error event
            print(f"Error: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            raise e

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
