from inference.rag_inference import RagInference, default_conversation_template
from dotenv import load_dotenv
from langchain.globals import set_debug, set_verbose
from langsmith import tracing_context

load_dotenv()

# less verbose than debug
set_verbose(True)
# set_debug(True)


class ChatRagInference(RagInference):
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
        self.retrieval_chain = self.compile(
            conversation_template=default_conversation_template
        )
        self.chat_history: list[tuple[str, str]] = []

    def query(self, query: str) -> dict:
        result = self.retrieval_chain.invoke(
            {"query": query, "chat_history": self.chat_history}
        )

        # Update chat history
        self.chat_history.append(("human", query))
        self.chat_history.append(("assistant", result["answer"]))

        return result


# CLI app for manual testing
if __name__ == "__main__":
    with tracing_context(enabled=True, project_name="langchain-pepwave"):
        rag_inference = ChatRagInference(
            llm_model="gpt-4.1-mini",
            pinecone_index_name="pepwave-early-april-page-content-embedding",
        )

        while True:
            query = input("\n\n*** Enter a query (or 'exit' to exit): ")

            if query.lower() == "exit":
                break

            result = rag_inference.query(query)
            print(f"\n\nAssistant: {result['answer']}\n")
