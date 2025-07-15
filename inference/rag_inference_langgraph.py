import operator
from typing import Annotated, Hashable
from langchain_core.documents import Document
from langchain_core.prompts import BasePromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.constants import START, END
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, InjectedState
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
INIT_RETRIEVAL = "NODE__INIT_RETRIEVAL"
PROMPT_LLM_W_TOOLS = "NODE__PROMPT_LLM_W_TOOLS"
GENERATE_RERANKER_QUERY = "NODE__GENERATE_RERANKER_QUERY"
TOOL_NODE = "NODE__TOOL_NODE"
HANDLE_TOOL_RESULTS = "NODE__HANDLE_TOOL_RESULTS"
RERANK = "NODE__RERANK"


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
        **kwargs,
    ):
        super().__init__(
            llm_model=llm_model, pinecone_index_name=pinecone_index_name, **kwargs
        )

        self.pinecone = PineconeRetriever(
            index_name=pinecone_index_name, embedding_model=self.embedding_model
        )

        self.checkpointer = checkpointer or InMemorySaver()

        self.tools = [self.semantic_search, self.search_web, self.search_wikipedia]

    def _entry_point_condition(self, state: MainState) -> Hashable | list[Hashable]:
        """
        If we have any context, start running the ReAct flow. Otherwise, initialize the first retrieval.
        """
        if state.current_context:
            return PROMPT_LLM_W_TOOLS
        else:
            return INIT_RETRIEVAL

    def _init_retrieval(self, state: MainState):
        messages = [
            ("system", PROMPTS["inference/retrieval_system"]),
            ("human", PROMPTS["inference/init_retrieval"].format(query=state.query)),
        ]
        llm = self.llm.bind_tools(self.tools, tool_choice="required")
        ai_tool_calls = llm.invoke(messages)
        return {"messages": [state.query, ai_tool_calls]}

    def _get_user_message(self, num_research_iterations: int) -> str:
        if num_research_iterations == 0:
            return PROMPTS["inference/user_new_followup"]
        if num_research_iterations > 2:
            return PROMPTS["inference/user_must_answer"]
        else:
            return PROMPTS["inference/user_retry_after_tools"]

    def _prompt_llm_w_tools(self, state: MainState) -> dict:
        allow_research = state.num_research_iterations > 2
        system_message = (
            PROMPTS["inference/system_allow_research"]
            if allow_research
            else PROMPTS["inference/system_deny_research"]
        )

        context = "\n\n</ContextDocument>\n\n<ContextDocument>\n\n".join(
            [doc.page_content for doc in state.current_context]
        )

        messages = [
            ("system", system_message.format(context=context)),
            ("placeholder", state.messages),
            (
                "human",
                self._get_user_message(state.num_research_iterations).format(
                    {"user_query": state.query}
                ),
            ),
        ]

        llm = (
            self.llm.bind_tools(self.tools, tool_choice="auto")
            if allow_research
            else self.llm
        )
        answer_or_tool_calls = llm.invoke(messages)
        return {"messages": [state.query, answer_or_tool_calls]}

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
        llm = self.llm.bind(temperature=0)
        messages = [
            ("system", PROMPTS["inference/generate_reranker_query_system"]),
            ("placeholder", state.messages),
        ]
        return {"reranker_query": llm.invoke(messages)}

    # "tools": ToolNode

    def _handle_tool_results(self, state: MainState):
        tool_results = state.latest_tool_call_results

        modified_messages = []
        for result in tool_results:
            if isinstance(result, ToolMessage):
                modified_message = ToolMessage(
                    content="Tool call successful, results appended to context corpus",
                    tool_call_id=result.tool_call_id,
                    name=result.name,
                )
                modified_messages.append(modified_message)

        return {
            "messages": modified_messages,
            "all_retrieved_context": tool_results,
        }

    def _rerank(self, state: MainState):
        return {"num_research_iterations": state.num_research_iterations + 1}

    @tool
    def semantic_search(
        self,
        search_query: Annotated[
            str,
            "The search query to use for semantic search in the vector database. Should be a well-formed query that describes what information you are looking for.",
        ],
    ) -> list[Document]:
        """The primary data source for information about Peplink products and services.
        Search the vector database for information relevant to the user query by providing
        a semantic search query. This tool is the primary
        means of retrieving information about Peplink products and services but it also
        contains some information about general IT networking concepts that are adjacent
        to Peplink products and services. You should always make at least one call to this
        too when doing research, typically with a version of the entire user query formatted
        for optimal semantic search via vector database.
        """
        query_embedding = self.pinecone.get_query_embedding(search_query)
        return self.pinecone.retrieve(
            search_query, query_embedding, top_k=70, rerank_top_n=20
        )

    @tool
    def search_web(
        self,
        search_query: Annotated[
            str,
            "A web search for a concept or entity to drill down on",
        ],
    ) -> list[Document]:
        """Use this tool when you need to drill down on a specific aspect of the user query by performing
        a web search using Google. This tool is most useful for general questions about IT networking and
        not for specific questions about Peplink products and services.
        """
        return [
            Document(
                page_content="Example web search results",
                metadata={
                    "source": "web",
                    "search_query": search_query,
                },
            )
        ]

    @tool
    def search_wikipedia(
        self,
        search_query: Annotated[
            str,
            "The search query to use for Wikipedia search. Should be a specific entity or concept.",
        ],
    ) -> list[Document]:
        """Use this tool to perform a Wikipedia search when you need general information about a topic that
        is not about Peplink/Pepwave products and services. This can be used to get information specific to
        the IT networking domain or information from any other domain in general. It is most useful for
        broad, general concepts that you would typically find in an encyclopedia.
        """
        return [
            Document(
                page_content="Example Wikipedia search results",
                metadata={
                    "source": "wikipedia",
                    "search_query": search_query,
                },
            )
        ]

    def compile(self) -> CompiledStateGraph:

        graph_builder = StateGraph(MainState)

        # Nodes
        graph_builder.add_node(INIT_RETRIEVAL, self._init_retrieval)
        graph_builder.add_node(PROMPT_LLM_W_TOOLS, self._prompt_llm_w_tools)
        graph_builder.add_node(GENERATE_RERANKER_QUERY, self._generate_reranker_query)
        graph_builder.add_node(
            TOOL_NODE,
            ToolNode(tools=self.tools, messages_key="latest_tool_call_results"),
        )
        graph_builder.add_node(HANDLE_TOOL_RESULTS, self._handle_tool_results)
        graph_builder.add_node(RERANK, self._rerank)

        # Edges
        graph_builder.set_conditional_entry_point(
            self._entry_point_condition,
            [INIT_RETRIEVAL, PROMPT_LLM_W_TOOLS],
        )
        graph_builder.add_edge(INIT_RETRIEVAL, TOOL_NODE)
        graph_builder.add_conditional_edges(
            PROMPT_LLM_W_TOOLS,
            self.tools_condition,
            [TOOL_NODE, GENERATE_RERANKER_QUERY, END],
        )
        graph_builder.add_edge(TOOL_NODE, HANDLE_TOOL_RESULTS)
        graph_builder.add_edge(GENERATE_RERANKER_QUERY, HANDLE_TOOL_RESULTS)
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
        )
