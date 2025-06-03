from inference.rag_inference_langgraph import RagInferenceLangGraph
from inference.rag_inference import default_conversation_template

from dotenv import load_dotenv
from langchain.globals import set_verbose
from langsmith import tracing_context
from datetime import datetime
import time

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
        self.graph = self.compile(conversation_template=default_conversation_template)

        # Only track current thread ID - no need for active_threads dictionary
        self.current_thread_id: str | None = None

    def create_new_thread(self) -> str:
        """Create a new conversation thread and return its ID."""
        # Generate a unique thread ID
        thread_id = f"thread_{int(time.time())}_{id(self)}"
        return thread_id


    def get_thread_history(self, thread_id: str | None = None) -> list:
        """Get the conversation history for a specific thread."""
        if thread_id is None:
            thread_id = self.current_thread_id

        try:
            state = self.graph.get_state(
                config={"configurable": {"thread_id": thread_id}}
            )
            return state.values.get("messages", []) if state.values else []
        except:
            return []

    def get_thread_message_count(self, thread_id: str | None = None) -> int:
        """Get the message count for a specific thread from LangGraph state."""
        messages = self.get_thread_history(thread_id)
        return len(messages)

    def _get_thread_title(self, thread_id: str) -> str:
        """Generate a title for a thread based on its first user message."""
        messages = self.get_thread_history(thread_id)
        if messages:
            for msg in messages:
                if hasattr(msg, 'type') and msg.type == "human":
                    return msg.content[:50] + ("..." if len(msg.content) > 50 else "")
        return "New Conversation"

    def _get_thread_created_at(self, thread_id: str) -> datetime:
        """Get the creation time of a thread from its earliest checkpoint."""
        if not self.checkpointer:
            return datetime.now()

        try:
            # Get the thread's state history and find the earliest checkpoint
            history = list(self.graph.get_state_history(
                config={"configurable": {"thread_id": thread_id}}
            ))
            if history:
                # History is ordered newest to oldest, so take the last one
                earliest_checkpoint = history[-1]
                if hasattr(earliest_checkpoint, 'created_at') and earliest_checkpoint.created_at:
                    created_at = earliest_checkpoint.created_at
                    if isinstance(created_at, str):
                        return datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    return created_at
        except Exception:
            pass
        return datetime.now()

    def list_threads(self) -> dict[str, dict]:
        """List all available threads by querying the checkpointer."""
        if not self.checkpointer:
            return {}

        try:
            # Get all checkpoints from the database
            checkpoints = list(self.checkpointer.list({}))

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
                messages = self.get_thread_history(thread_id)
                if messages:  # Only include threads with messages
                    threads[thread_id] = {
                        "created_at": self._get_thread_created_at(thread_id),
                        "title": self._get_thread_title(thread_id),
                    }

            return threads

        except Exception as e:
            print(f"⚠️ Warning: Could not list threads from database: {e}")
            return {}

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a conversation thread and its associated state."""
        if not self.checkpointer:
            return False

        try:
            # Use the checkpointer's built-in delete_thread method
            # This properly deletes all checkpoints and writes from the database
            self.checkpointer.delete_thread(thread_id)

            # If we deleted the current thread, clear current thread
            if thread_id == self.current_thread_id:
                self.current_thread_id = None

            return True

        except Exception:
            raise KeyError(f"Thread {thread_id} not found")

    def query(self, query: str, thread_id: str | None = None):
        """Stream the response token by token using LangGraph's messages streaming mode."""
        if thread_id is None:
            thread_id = self.current_thread_id

        initial_state = {"query": query, "thread_id": thread_id}

        # Use stream with messages mode to get token-by-token streaming
        for chunk in self.graph.stream(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="messages",
        ):
            # chunk is a tuple of (message_chunk, metadata)
            if isinstance(chunk, tuple) and len(chunk) == 2:
                message_chunk, metadata = chunk
                # Only stream content from the generate_answer node (the LLM response)
                if (
                    hasattr(message_chunk, 'content')
                    and message_chunk.content
                    and metadata.get('langgraph_node') == 'generate_answer'
                ):
                    yield str(message_chunk.content)
