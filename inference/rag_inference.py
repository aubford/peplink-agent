from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from config import global_config
from langchain import hub
from pinecone import Pinecone
from langchain_core.runnables.passthrough import RunnablePassthrough
from inference.history_aware_retrieval_query import history_aware_retrieval_query

# Note: for reasoning models: "include only the most relevant information to prevent the model from overcomplicating its response." - api docs
# Other advice for reasoning models: https://platform.openai.com/docs/guides/reasoning#advice-on-prompting


class RagInference:
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-large",
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0,
        streaming: bool = False,
    ):
        # Initialize Pinecone
        pinecone = Pinecone(api_key=global_config.get("PINECONE_API_KEY"))
        self.index = pinecone.Index(global_config.get("VERSIONED_PINECONE_INDEX_NAME"))

        # Initialize components
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = PineconeVectorStore(
            index=self.index, embedding=self.embeddings, text_key="page_content"
        )

        self.llm = ChatOpenAI(
            model=llm_model, temperature=temperature, streaming=streaming
        )
        self.prompt = hub.pull("aubford/retrieval-qa-chat")

        # Setup retriever
        self.retriever = (
            lambda x: x["retrieval_query"]
        ) | self.vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 20, "fetch_k": 50}
        )

        # Setup chain
        self.retrieval_chain = (
            RunnablePassthrough.assign(retrieval_query=history_aware_retrieval_query())
            .assign(
                context=self.retriever.with_config(run_name="retrieve_documents"),
            )
            .assign(answer=create_stuff_documents_chain(self.llm, self.prompt))
        ).with_config(run_name="rag_inference")

        # Initialize chat history
        self.chat_history: list[tuple[str, str]] = []

    def query(self, query: str) -> dict:
        """
        Process a single query through the RAG pipeline.

        Args:
            query: The user's question

        Returns:
            dict: Contains the answer and other chain outputs
        """
        result = self.retrieval_chain.invoke(
            {"input": query, "chat_history": self.chat_history}
        )

        # Update chat history
        self.chat_history.append(("human", query))
        self.chat_history.append(("assistant", result["answer"]))

        return result

    def clear_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []


if __name__ == "__main__":
    rag_inference = RagInference(streaming=True)

    while True:
        # Get user input
        query = input("\n\n*** Enter a query (or 'exit' to exit): ")

        # Check for exit condition
        if query.lower() == "exit":
            break

        result = rag_inference.query(query)
        print(f"\n\nAssistant: {result['answer']}\n")
