from inference.rag_inference_langgraph import RagInferenceLangGraph
from rag_inference import default_conversation_template

from dotenv import load_dotenv
from langchain.globals import set_verbose
from langsmith import tracing_context

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

    def query(self, query: str, thread_id: str) -> dict:
        initial_state = {"query": query, "thread_id": thread_id}
        result = self.graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )
        return result

    def stream_query(self, query: str, thread_id: str):
        """Stream the response token by token using LangGraph's messages streaming mode."""
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

    async def astream_query(self, query: str, thread_id: str):
        """Async version of stream_query for token-by-token streaming."""
        initial_state = {"query": query, "thread_id": thread_id}

        # Use astream with messages mode to get token-by-token streaming
        async for chunk in self.graph.astream(
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


thread_id = "single_thread"
if __name__ == "__main__":
    with tracing_context(enabled=True, project_name="langchain-pepwave"):
        rag_inference = ChatLangGraph(
            llm_model="gpt-4.1-mini",
            pinecone_index_name="pepwave-early-april-page-content-embedding",
        )

        while True:
            query = input("\n\n*** Enter a query (or 'exit' to exit): ")

            if query.lower() == "exit":
                break

            # Demonstrate streaming
            print("\n\nAssistant (streaming): ", end="", flush=True)
            for token in rag_inference.stream_query(query, thread_id):
                print(token, end="", flush=True)
            print("\n")  # New line after streaming is complete
