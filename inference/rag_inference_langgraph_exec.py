from inference.rag_inference_langgraph import RagInferenceLangGraph
from rag_inference import default_conversation_template

from dotenv import load_dotenv
from langchain.globals import set_debug, set_verbose
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
        minimal_tracer: bool = False,
    ):
        super().__init__(
            llm_model,
            pinecone_index_name,
            minimal_tracer=minimal_tracer,
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

            result = rag_inference.query(query, thread_id)
            print(f"\n\nAssistant: {result['answer']}\n")
