from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.checkpoint.memory import InMemorySaver
from inference.rag_inference_langgraph import RagInferenceLangGraph, PROMPT_LLM_W_TOOLS

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

    def get_thread_message_count(self, thread_id: str) -> int:
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
            history = list(
                self.graph.get_state_history(
                    config={"configurable": {"thread_id": thread_id}}
                )
            )
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
            return True

        except Exception:
            raise KeyError(f"Thread {thread_id} not found")

    def get_thread_history(self, thread_id: str) -> list:
        """Get the conversation history for a specific thread."""
        state = self.graph.get_state(config={"configurable": {"thread_id": thread_id}})
        return state.values.get("messages", []) if state.values else []

    def query(self, query: str, thread_id: str):
        """Stream the response token by token using LangGraph's messages streaming mode."""
        for stream_mode, chunk in self.graph.stream(
            {"query": query, "thread_id": thread_id},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode=[
                "values",
                "messages",
            ],
        ):
            if stream_mode == "values":
                yield chunk["messages"]  # type: ignore
            # chunk is a tuple of (message_chunk, metadata)
            if isinstance(chunk, tuple) and len(chunk) == 2:
                message_chunk, metadata = chunk
                # Only stream content from the generate_answer node (the LLM response)
                if (
                    metadata.get('langgraph_node') == PROMPT_LLM_W_TOOLS
                    and message_chunk.text()
                ):
                    yield str(message_chunk.text())

    def draw_graph(self):
        self.graph.get_graph().print_ascii()
        self.graph.get_graph(xray=True).draw_mermaid_png(
            output_file_path="graph_diagram.png",
            draw_method=MermaidDrawMethod.PYPPETEER,
            max_retries=3,
            retry_delay=2.0,
        )


if __name__ == "__main__":
    chatbot = ChatLangGraph(
        llm_model="gpt-4.1",
        pinecone_index_name="pepwave-early-april-page-content-embedding",
        checkpointer=InMemorySaver(),
    )
    chatbot.draw_graph()
