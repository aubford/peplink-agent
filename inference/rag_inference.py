from typing import Any
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables.passthrough import RunnablePassthrough
from inference.history_aware_retrieval_query import (
    get_history_aware_retrieval_query_chain,
)
from langchain_core.language_models.chat_models import BaseChatModel
from util.root_only_tracer import RootOnlyTracer
from prompts import load_prompts
from langchain_core.runnables import RunnableConfig
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from inference.cohere_rerank import RateLimitedCohereRerank
from inference.rate_limiters import openai_rate_limiter

# Note: for reasoning models: "include only the most relevant information to prevent the model from overcomplicating its response." - api docs
# Other advice for reasoning models: https://platform.openai.com/docs/guides/reasoning#advice-on-prompting

PROMPTS = load_prompts()

messages_prompt = ChatPromptTemplate(
    [
        ("system", PROMPTS['inference/system']),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)


class RagInference:
    def __init__(
        self,
        llm_model: str,
        pinecone_index_name: str,
        embedding_model: str = "text-embedding-3-large",
        temperature: float = 1,  # openai default temp
        streaming: bool = False,
        eval_llm: BaseChatModel | None = None,
        messages: ChatPromptTemplate = messages_prompt,
        minimal_tracer: bool = False,
    ):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = PineconeVectorStore(
            index_name=pinecone_index_name,
            embedding=self.embeddings,
            text_key="page_content",
        )

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            streaming=streaming,
            rate_limiter=openai_rate_limiter,
        )

        retriever = self.vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 30, "fetch_k": 50}
        )

        compressor = RateLimitedCohereRerank(model="rerank-v3.5", top_n=20)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        config = RunnableConfig({"run_name": "rag_inference"})
        if minimal_tracer:
            config["callbacks"] = [RootOnlyTracer(project_name="langchain-pepwave")]

        self.retrieval_chain = (
            RunnablePassthrough.assign(
                retrieval_query=get_history_aware_retrieval_query_chain(llm=self.llm)
            )
            .assign(
                context=(lambda x: x["retrieval_query"]) | self.retriever,
            )
            .assign(
                answer=create_stuff_documents_chain(
                    eval_llm or self.llm,
                    messages,
                    document_separator="\n\n</ContextDocument>\n\n<ContextDocument>\n\n",
                )
            )
        ).with_config(config)

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
            {"query_id": query_id, "input": query, "chat_history": []}
            for query_id, query in queries.items()
        ]

        # Use native batch processing with proper rate limiting
        results = await self.retrieval_chain.abatch(
            batch_inputs, config={"max_concurrency": 20}
        )

        for result in results:
            result["custom_id"] = result["answer"]
        return {result["query_id"]: result for result in results}
