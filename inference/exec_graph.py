from inference.rag_inference_langgraph import RagInferenceLangGraph
from rag_inference import default_conversation_template

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
    ):
        super().__init__(
            llm_model,
            pinecone_index_name,
            minimal_tracer=minimal_tracer,
            temperature=temperature,
            streaming=True,
        )
        self.graph = self.compile(conversation_template=default_conversation_template)

        # Thread management
        self.active_threads: dict[str, dict] = {}
        self.current_thread_id: str = self.create_new_thread()

    def create_new_thread(self) -> str:
        """Create a new conversation thread and return its ID."""
        thread_id = f"thread_{len(self.active_threads) + 1}_{int(time.time())}"
        self.active_threads[thread_id] = {
            "created_at": datetime.now(),
            "title": "New Conversation",
        }
        return thread_id

    def switch_to_thread(self, thread_id: str) -> bool:
        """Switch to an existing thread."""
        if thread_id in self.active_threads:
            self.current_thread_id = thread_id
            return True
        return False

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

    def list_threads(self) -> dict[str, dict]:
        """List all available threads."""
        return self.active_threads

    def query(self, query: str, thread_id: str | None = None):
        """Stream the response token by token using LangGraph's messages streaming mode."""
        if thread_id is None:
            thread_id = self.current_thread_id

        # Update thread metadata
        if thread_id in self.active_threads:
            if self.active_threads[thread_id]["title"] == "New Conversation":
                # Auto-generate title from first query
                self.active_threads[thread_id]["title"] = query[:50] + (
                    "..." if len(query) > 50 else ""
                )

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

    def print_help(self):
        """Print available commands."""
        print("\n=== Available Commands ===")
        print("/new - Create a new conversation thread")
        print("/list - List all conversation threads")
        print("/switch <thread_id> - Switch to a specific thread")
        print("/history - Show current thread's conversation history")
        print("/help - Show this help message")
        print("/exit - Exit the application")
        print("========================\n")


if __name__ == "__main__":
    with tracing_context(enabled=True, project_name="langchain-pepwave"):
        rag_inference = ChatLangGraph(
            llm_model="gpt-4.1-mini",
            pinecone_index_name="pepwave-early-april-page-content-embedding",
        )

        print("🤖 ChatLangGraph with Thread Management")
        print(f"Current thread: {rag_inference.current_thread_id}")
        rag_inference.print_help()

        while True:
            query = input(
                f"\n[{rag_inference.current_thread_id}] Enter query or command: "
            )

            if query.startswith("/exit"):
                break
            elif query.startswith("/help"):
                rag_inference.print_help()
            elif query.startswith("/new"):
                new_thread = rag_inference.create_new_thread()
                rag_inference.switch_to_thread(new_thread)
                print(f"✅ Created and switched to new thread: {new_thread}")
            elif query.startswith("/list"):
                threads = rag_inference.list_threads()
                print("\n📋 Available Threads:")
                for tid, info in threads.items():
                    current_marker = (
                        "👉 " if tid == rag_inference.current_thread_id else "   "
                    )
                    print(
                        f"{current_marker}{tid}: {info['title']} ({rag_inference.get_thread_message_count(tid)} messages)"
                    )
                print()
            elif query.startswith("/switch"):
                thread_id = query[8:].strip()
                if not thread_id:
                    print("❌ Please provide a thread ID. Usage: /switch <thread_id>")
                    print("💡 Use /list to see available threads")
                elif rag_inference.switch_to_thread(thread_id):
                    print(f"✅ Switched to thread: {thread_id}")
                else:
                    print(f"❌ Thread not found: {thread_id}")
            elif query.startswith("/history"):
                history = rag_inference.get_thread_history()
                if history:
                    print(f"\n📜 History for {rag_inference.current_thread_id}:")
                    for i, msg in enumerate(history):
                        role = "🧑 User" if msg.type == "human" else "🤖 Assistant"
                        content = msg.content[:100] + (
                            "..." if len(msg.content) > 100 else ""
                        )
                        print(f"  {i+1}. {role}: {content}")
                else:
                    print("📜 No conversation history for this thread.")
            elif query.strip():
                # Regular query
                print(f"\n🤖 Assistant: ", end="", flush=True)
                for token in rag_inference.query(query):
                    print(token, end="", flush=True)
                print("\n")
            else:
                print(
                    "Please enter a query or command. Type /help for available commands."
                )
