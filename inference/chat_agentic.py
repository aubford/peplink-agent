import json
from typing import AsyncGenerator
from langchain_core.messages import HumanMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from inference.rag_inference_langgraph import RagInferenceLangGraph, StreamMessage

from dotenv import load_dotenv
from langchain.globals import set_verbose
from datetime import datetime

# from langsmith import tracing_context

load_dotenv()

# less verbose than debug
set_verbose(True)
# set_debug(True)


class ChatLangGraph(RagInferenceLangGraph):
    def __init__(
        self,
        llm_model: str,
        pinecone_index_name: str,
        temperature: float = 1,
        minimal_tracer: bool = False,
        checkpointer=None,
    ):
        super().__init__(
            llm_model,
            pinecone_index_name,
            minimal_tracer=minimal_tracer,
            temperature=temperature,
            checkpointer=checkpointer,
            streaming=True,
        )
        self.graph = self.compile()

    async def get_thread_message_count(self, thread_id: str) -> int:
        """Get the message count for a specific thread from LangGraph state."""
        messages = await self.get_thread_history(thread_id)
        return len(messages)

    async def _get_thread_title(self, thread_id: str) -> str:
        """Generate a title for a thread based on its first user message."""
        messages = await self.get_thread_history(thread_id)
        if messages:
            for msg in messages:
                if hasattr(msg, 'type') and msg.type == "human":
                    return msg.content[:50] + ("..." if len(msg.content) > 50 else "")
        return "New Conversation"

    async def _get_thread_created_at(self, thread_id: str) -> datetime:
        """Get the creation time of a thread from its earliest checkpoint."""
        if not self.checkpointer:
            return datetime.now()

        try:
            # Get the thread's state history and find the earliest checkpoint
            history = [
                cp
                async for cp in self.graph.aget_state_history(
                    config={"configurable": {"thread_id": thread_id}}
                )
            ]
            if history:
                # History is ordered newest to oldest, so take the last one
                earliest_checkpoint = history[-1]
                if (
                    hasattr(earliest_checkpoint, 'created_at')
                    and earliest_checkpoint.created_at
                ):
                    created_at = earliest_checkpoint.created_at
                    if isinstance(created_at, str):
                        return datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    return created_at
        except Exception:
            pass
        return datetime.now()

    async def list_threads(self) -> dict[str, dict]:
        """List all available threads by querying the checkpointer."""
        if not self.checkpointer:
            return {}

        try:
            # Get all checkpoints from the database
            checkpoints = [cp async for cp in self.checkpointer.alist(None)]

            # Extract unique thread IDs
            thread_ids = set()
            for checkpoint in checkpoints:
                thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
                if thread_id:
                    thread_ids.add(thread_id)

            # Build thread info for each unique thread
            threads = {}
            for thread_id in thread_ids:
                # Only include threads that have actual conversation messages
                # TODO: Clean up all these redundant database calls via get_thread_history.
                messages = await self.get_thread_history(thread_id)
                if messages:  # Only include threads with messages
                    threads[thread_id] = {
                        "created_at": await self._get_thread_created_at(thread_id),
                        "title": await self._get_thread_title(thread_id),
                    }

            return threads

        except Exception as e:
            print(f"⚠️ Warning: Could not list threads from database: {e}")
            return {}

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a conversation thread and its associated state."""
        if not self.checkpointer:
            return False

        try:
            # Use the checkpointer's built-in delete_thread method
            # This properly deletes all checkpoints and writes from the database
            await self.checkpointer.adelete_thread(thread_id)
            return True

        except Exception:
            raise KeyError(f"Thread {thread_id} not found")

    async def get_thread_history(self, thread_id: str) -> list:
        """Get the conversation history for a specific thread."""
        state = await self.graph.aget_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        return state.values.get("messages", []) if state.values else []

    async def query(self, user_query: str, thread_id: str) -> AsyncGenerator[str, None]:
        """Stream the response using LangGraph's messages streaming mode."""
        async for stream_mode, chunk in self.graph.astream(
            {"messages": [HumanMessage(content=user_query)], "thread_id": thread_id},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode=[
                "values",
                "custom",
            ],
        ):
            if isinstance(chunk, dict) and stream_mode == "values":
                messages = [
                    {"type": msg.type, "content": msg.content}
                    for msg in chunk["messages"]
                    if msg.content and msg.content.strip()
                ]
                yield json.dumps({'type': 'messages', 'messages': messages})
            elif isinstance(chunk, StreamMessage):
                yield chunk.model_dump_json()

    def draw_graph(self):
        self.graph.get_graph().print_ascii()
        self.graph.get_graph(xray=True).draw_mermaid_png(
            output_file_path="graph_diagram.png",
            draw_method=MermaidDrawMethod.PYPPETEER,
            max_retries=3,
            retry_delay=2.0,
        )
