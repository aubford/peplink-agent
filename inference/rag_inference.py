import httpx
from abc import ABC
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from inference.logging_transport import LoggingTransport, AsyncLoggingTransport
from util.root_only_tracer import RootOnlyTracer
from prompts import load_prompts
from langchain_core.runnables import RunnableConfig

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
        temperature: float = 0.5,  # openai default temp
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
        self.client_with_logging = httpx.Client(
            transport=LoggingTransport(httpx.HTTPTransport())
        )
        self.async_client_with_logging = httpx.AsyncClient(
            transport=AsyncLoggingTransport(httpx.AsyncHTTPTransport())
        )

        self.config = RunnableConfig({"run_name": "rag_inference"})
        if minimal_tracer:
            self.config["callbacks"] = [
                RootOnlyTracer(project_name="langchain-pepwave")
            ]

    def set_temperature(self, temperature: float):
        self.temperature = temperature
