import operator
from typing import Annotated, Hashable
from langchain_core.prompts import BasePromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.constants import START, END
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from inference.cohere_rerank import RateLimitedCohereRerank
from inference.rag_inference import InferenceBase
from inference.pinecone_retriever import PineconeRetriever
from inference.rate_limiters import openai_rate_limiter
from prompts import load_prompts


from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from load.batch_manager import BatchManager
from evals.batch_llm import BatchChatOpenAI
from pydantic import BaseModel, Field
from typing_extensions import Literal


# Load prompts
PROMPTS = load_prompts()

# Node name constants
EMBED_QUERY = "NODE__EMBED_QUERY"
INIT_RETRIEVAL = "NODE__INIT_RETRIEVAL"
PROMPT_LLM_W_TOOLS = "NODE__PROMPT_LLM_W_TOOLS"
GENERATE_RERANKER_QUERY = "NODE__GENERATE_RERANKER_QUERY"
TOOL_NODE = "NODE__TOOL_NODE"
HANDLE_TOOL_RESULTS = "NODE__HANDLE_TOOL_RESULTS"
RERANK = "NODE__RERANK"


default_conversation_template = ChatPromptTemplate(
    [
        ("system", PROMPTS["inference/system"]),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
    ]
)


# Define the state schema using Pydantic
class MainState(BaseModel):
    """State for RAG inference system using Pydantic.

    Attributes:
        tool_call_results: only used for intra-graph operations
        all_retrieved_context: used for future reranking
        current_context: the actual docs exposed to the LLM in system msg
        messages: What the user sees
        query: current user query
        num_research_iterations: number of times research has been done for a single user query
        reranker_query: pretty much only used for tracing
        reranker_query_embedding: used for reranking in following step
        answer: the answer to the user query
        thread_id: the thread id for the user query
    """

    messages: Annotated[list, add_messages] = Field(default_factory=list)
    num_research_iterations: int = 0
    query: str = ""
    query_embedding: list[float] = Field(default_factory=list)
    current_context: list = Field(default_factory=list)
    all_retrieved_context: Annotated[list, operator.add] = Field(default_factory=list)
    thread_id: str = "default"
    latest_tool_call_results: list = Field(default_factory=list)
    reranker_query: str = ""
    reranker_query_embedding: list[float] = Field(default_factory=list)
    tool_call_results: list = Field(default_factory=list)

    answer: str = ""


class RagInferenceLangGraph(InferenceBase):
    """
    Contains the logic for running the research and question answering agent pair. The goal is
    to perform thorough research as quickly and efficiently as possible with the least number of
    round-trips to the model provider.
    """

    def __init__(
        self,
        llm_model: str,
        pinecone_index_name: str,
        checkpointer: BaseCheckpointSaver | None = None,
        conversation_template: BasePromptTemplate = default_conversation_template,
        **kwargs,
    ):
        super().__init__(
            llm_model=llm_model, pinecone_index_name=pinecone_index_name, **kwargs
        )

        self.pinecone = PineconeRetriever(
            index_name=pinecone_index_name, embedding_model=self.embedding_model
        )

        self.checkpointer = checkpointer or InMemorySaver()
        self.conversation_template = conversation_template

        self.tools = []


    def _entry_point_condition(self, state: MainState) -> Hashable | list[Hashable]:
        """
        If we have any context, start running the ReAct flow. Otherwise, initialize the first retrieval.
        """
        if state.current_context:
            return PROMPT_LLM_W_TOOLS
        else:
            return [EMBED_QUERY, INIT_RETRIEVAL]

    def _embed_query(self, state: MainState):
        query_embedding = self.pinecone.get_query_embedding(state.query)
        return {"query_embedding": query_embedding}

    def _init_retrieval(self, state: MainState):
        return {}

    def _prompt_llm_w_tools(self, state: MainState) -> dict:
        """Generate an answer based on the context and query."""
        messages = self.conversation_template.invoke(
            {
                "query": state.query,
                "chat_history": state.messages,
                "context": "\n\n</ContextDocument>\n\n<ContextDocument>\n\n".join(
                    [doc.page_content for doc in state.current_context]
                ),
            }
        )
        answer = self.llm.invoke(messages)
        return {"messages": [answer]}

    def tools_condition(self, state: MainState) -> Hashable | list[Hashable]:
        """
        Determine if we should run tools or exit the flow.
        """
        ai_message = state.messages[-1]
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return [TOOL_NODE, GENERATE_RERANKER_QUERY]
        else:
            return END

    def _generate_reranker_query(self, state: MainState):
        """Generate a reranker query considering chat history."""
        return {}

    # "tools": ToolNode

    def _handle_tool_results(self, state: MainState):
        return {}

    def _rerank(self, state: MainState):
        return {}

    def _semantic_search_tool(self, state: MainState) -> dict:
        """Retrieve relevant documents based on the query."""
        assert state.query_embedding
        retrieved_context = self.pinecone.retrieve(
            state.query, state.query_embedding, top_k=100, rerank_top_n=40
        )

        return {
            "context": retrieved_context,
            "context_history": retrieved_context,
        }


    def compile(self) -> CompiledStateGraph:

        graph_builder = StateGraph(MainState)

        # Nodes
        graph_builder.add_node(EMBED_QUERY, self._embed_query)
        graph_builder.add_node(INIT_RETRIEVAL, self._init_retrieval)
        graph_builder.add_node(PROMPT_LLM_W_TOOLS, self._prompt_llm_w_tools)
        graph_builder.add_node(GENERATE_RERANKER_QUERY, self._generate_reranker_query)
        graph_builder.add_node(TOOL_NODE, ToolNode(tools=self.tools, messages_key="tool_call_results"))
        graph_builder.add_node(HANDLE_TOOL_RESULTS, self._handle_tool_results)
        graph_builder.add_node(RERANK, self._rerank)

        # Edges
        graph_builder.set_conditional_entry_point(self._entry_point_condition)
        graph_builder.add_edge(EMBED_QUERY, TOOL_NODE)
        graph_builder.add_edge(INIT_RETRIEVAL, TOOL_NODE)
        graph_builder.add_conditional_edges(
            PROMPT_LLM_W_TOOLS,
            self.tools_condition,
        )
        graph_builder.add_edge(TOOL_NODE, HANDLE_TOOL_RESULTS)
        graph_builder.add_edge(HANDLE_TOOL_RESULTS, RERANK)
        graph_builder.add_edge(RERANK, PROMPT_LLM_W_TOOLS)

        # Compile
        compiled_graph = graph_builder.compile(checkpointer=self.checkpointer)
        graph = compiled_graph.with_config(self.config)
        return graph


    def _get_cohere_retriever(self) -> ContextualCompressionRetriever:
        retriever_base = self.vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 60, "fetch_k": 100}
        )
        compressor = RateLimitedCohereRerank(model="rerank-v3.5", top_n=40)
        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever_base
        )


    @property
    def llm(self):
        return ChatOpenAI(
            model=self.llm_model,
            temperature=self.temperature,
            streaming=self.streaming,
            rate_limiter=openai_rate_limiter,
            use_responses_api=True,
            output_version="responses/v1",
        ).bind_tools(self.tools)

