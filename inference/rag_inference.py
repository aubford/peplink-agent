from typing import Any
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from config import global_config
from langchain import hub
from langchain_core.runnables.passthrough import RunnablePassthrough
from inference.history_aware_retrieval_query import (
    get_history_aware_retrieval_query_chain,
)
from evals.batch_llm import BatchChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.language_models.chat_models import BaseChatModel
from util.root_only_tracer import RootOnlyTracer

# Note: for reasoning models: "include only the most relevant information to prevent the model from overcomplicating its response." - api docs
# Other advice for reasoning models: https://platform.openai.com/docs/guides/reasoning#advice-on-prompting

prompt = hub.pull("aubford/retrieval-qa-chat")


class RagInference:
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-large",
        llm_model: str = "gpt-4.1-mini",
        temperature: float = 0.2,
        streaming: bool = False,
        pinecone_index_name: str = global_config.get("VERSIONED_PINECONE_INDEX_NAME"),
        eval_llm: BaseChatModel | None = None,
    ):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = PineconeVectorStore(
            index_name=pinecone_index_name,
            embedding=self.embeddings,
            text_key="page_content",
            pinecone_api_key=global_config.get("PINECONE_API_KEY"),
        )

        # Create a rate limiter for the model to avoid API throttling
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=3.0,
            max_bucket_size=10,
        )

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            streaming=streaming,
            rate_limiter=rate_limiter,
        )

        self.prompt = prompt

        self.minimal_tracer = RootOnlyTracer(project_name="langchain-pepwave")

        self.retriever = (
            lambda x: x["retrieval_query"]
        ) | self.vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 20, "fetch_k": 50}
        )

        self.retrieval_chain = (
            RunnablePassthrough.assign(
                retrieval_query=get_history_aware_retrieval_query_chain(llm=self.llm)
            )
            .assign(
                context=self.retriever,
            )
            .assign(
                answer=create_stuff_documents_chain(
                    eval_llm or self.llm,
                    self.prompt,
                    document_separator="\n\n</ContextDocument>\n\n<ContextDocument>\n\n",
                )
            )
        ).with_config({"run_name": "rag_inference", "callbacks": [self.minimal_tracer]})

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

    async def batch_query_for_eval(
        self, queries: dict[str, str]
    ) -> dict[str, dict[str, Any]]:
        """
        Process multiple queries in parallel for evaluation purposes using LangChain's
        native batch processing capabilities.

        Args:
            queries: Dictionary of query identifiers to queries

        Returns:
            dict: Mapping of query identifiers to results
        """

        # Get just the inputs for processing
        batch_inputs = [
            {"input": query, "chat_history": []} for query in queries.values()
        ]

        # Use native batch processing with proper rate limiting
        results = await self.retrieval_chain.abatch(
            batch_inputs, config={"max_concurrency": 20}
        )

        key_to_hash_map = {
            key: BatchChatOpenAI.hash_messages(query) for key, query in queries.items()
        }
        # Recombine results with their keys
        recombined = {
            key: result for key, result in zip(key_to_hash_map.keys(), results)
        }
        for key, result in recombined.items():
            assert (
                result["answer"] == key_to_hash_map[key]
            ), "custom_id does not match key to hash map"
        return recombined


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
