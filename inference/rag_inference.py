from abc import ABC, abstractmethod
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables.passthrough import RunnablePassthrough
from inference.history_aware_retrieval_query import (
    get_history_aware_retrieval_chain,
)
from util.root_only_tracer import RootOnlyTracer
from load.batch_manager import BatchManager
from evals.batch_llm import BatchChatOpenAI
from prompts import load_prompts
from langchain_core.runnables import RunnableConfig
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from inference.cohere_rerank import RateLimitedCohereRerank
from langchain_core.runnables.base import Runnable

# Note: for reasoning models: "include only the most relevant information to prevent the model from overcomplicating its response." - api docs
# Other advice for reasoning models: https://platform.openai.com/docs/guides/reasoning#advice-on-prompting

PROMPTS = load_prompts()

default_conversation_template = ChatPromptTemplate(
    [
        ("system", PROMPTS["inference/system"]),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
    ]
)


class InferenceBase(ABC):
    def __init__(
        self,
        llm_model: str,
        pinecone_index_name: str,
        embedding_model: str = "text-embedding-3-large",
        temperature: float = 1,  # openai default temp
        streaming: bool = False,
        minimal_tracer: bool = False,
    ):
        self.embedding_model = embedding_model
        self.vector_store = PineconeVectorStore(
            index_name=pinecone_index_name,
            embedding=OpenAIEmbeddings(model=embedding_model),
            text_key="page_content",
        )
        self.llm_model = llm_model
        self.temperature = temperature
        self.streaming = streaming
        self.pinecone_index_name = pinecone_index_name

        self.config = RunnableConfig({"run_name": "rag_inference"})
        if minimal_tracer:
            self.config["callbacks"] = [
                RootOnlyTracer(project_name="langchain-pepwave")
            ]

    def set_temperature(self, temperature: float):
        self.temperature = temperature
