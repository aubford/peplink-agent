import re
from typing import Annotated, Hashable, Literal, TypedDict
from langchain_core.documents import Document
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.messages.ai import AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_core.messages import (
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    ToolMessage,
)
from langgraph.constants import END, START
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.config import get_stream_writer
from langchain_core.output_parsers import JsonOutputParser
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
from pydantic import BaseModel, Field


# Load prompts
PROMPTS = load_prompts()

# Node name constants
ENTRY_POINT = "NODE__ENTRY_POINT"
INIT_RETRIEVAL = "NODE__INIT_RETRIEVAL"
PROMPT_LLM_W_TOOLS = "NODE__PROMPT_LLM_W_TOOLS"
GENERATE_RERANKER_QUERY = "NODE__GENERATE_RERANKER_QUERY"
TOOL_NODE = "NODE__TOOL_NODE"
HANDLE_TOOL_RESULTS = "NODE__HANDLE_TOOL_RESULTS"
RERANK = "NODE__RERANK"
STREAM = "STREAM"
PASSTHRU = "PASSTHRU"


class StreamMessage(BaseModel):
    type: Literal["token", "messages", "complete", "error", "log"]
    content: str


# TODO: can remove?
class RAGInferenceOutputTyped(TypedDict):
    have_enough_information: Annotated[
        bool,
        ...,
        "Whether you have enough information to answer the user query using only the context corpus documents above.",
    ]
    have_enough_information_reasoning: Annotated[
        str,
        ...,
        "Briefly explain your reasoning for why you do or do not have enough information to answer the user query.",
    ]
    answer: Annotated[
        str,
        ...,
        "The answer to the user query. If you do not have enough information, you should return an empty string.",
    ]


class RAGInferenceOutput(BaseModel):
    have_enough_information: bool = Field(
        description="Whether you have enough information to answer the user query."
    )
    have_enough_information_reasoning: str = Field(
        description="Reasoning for why you do or do not have enough information to answer the user query."
    )
    answer: str = Field(
        description="The answer to the user query. If you do not have enough information, you should return an empty string."
    )


# Define the state schema using Pydantic
class MainState(BaseModel):
    """State for RAG inference system using Pydantic.

    Attributes:
        current_context: the actual docs exposed to the LLM in system msg
        messages: What the user sees
        current_turn_index: the index of the user query message in the messages list for the current turn
        num_research_iterations: number of times research has been done for a single user query
        reranker_query: pretty much only used for tracing
        thread_id: the thread id for the user query
    """

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    num_research_iterations: int = 0
    current_turn_query_index: int = 0
    current_context: list = Field(default_factory=list)
    reranker_query: str = ""
    thread_id: str = "default"


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

        self.tools = self._create_tools()

    def _stream_message(
        self,
        message_type: Literal["token", "messages", "complete", "error", "log"],
        content: str,
    ) -> None:
        """Helper to create and write a StreamMessage using the current stream writer."""
        try:
            writer = get_stream_writer()
            writer(StreamMessage(type=message_type, content=content))
        except Exception:
            # Silently ignore if no stream writer context is available
            raise RuntimeError("No stream writer context is available")

    def get_user_query(self, state: MainState) -> str:
        """Get the content of the user query message at current_turn_query_index."""
        message = state.messages[state.current_turn_query_index]
        return str(message.content)

    def node_entry_point(self, state: MainState):
        self._stream_message(
            "log",
            f"🔍 Entry point state: {state.model_dump_json(indent=2)}",
        )
        return {
            "num_research_iterations": 0,
            "current_turn_query_index": len(state.messages) - 1,
        }

    def _entry_point_condition(self, state: MainState) -> Hashable | list[Hashable]:
        """
        If we have any context, start running the ReAct flow. Otherwise, initialize the first retrieval.
        """
        if state.current_context:
            return PROMPT_LLM_W_TOOLS
        else:
            return INIT_RETRIEVAL

    def _format_response(
        self,
        ai_message: AIMessage,
        intro_text: str = "I'll research your question using these tools:",
    ) -> AIMessage:
        """Format the response from the LLM whether it be a tool call or the final answer."""
        if not ai_message.tool_calls:
            ai_message.content = ai_message.additional_kwargs["parsed"].answer
            return ai_message

        tool_descriptions = []
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call['name']
            if tool_name == 'semantic_search':
                tool_descriptions.append(
                    f"🔍 Searching vector database for: {tool_call['args']['search_query']}"
                )
            elif tool_name == 'search_web':
                tool_descriptions.append(
                    f"🌐 Searching web for: {tool_call['args']['search_query']}"
                )
            elif tool_name == 'search_wikipedia':
                tool_descriptions.append(
                    f"📚 Searching Wikipedia for: {tool_call['args']['search_query']}"
                )

        if tool_descriptions:
            ai_message.content = f"{intro_text}\n\n" + "\n\n".join(tool_descriptions)

        return ai_message

    def node_init_retrieval(self, state: MainState):
        messages = [
            ("system", PROMPTS["inference/retrieval_system"]),
            (
                "human",
                PROMPTS["inference/init_retrieval"].format(
                    user_query=self.get_user_query(state)
                ),
            ),
        ]
        llm = self.llm.bind_tools(self.tools, tool_choice="required")
        ai_tool_calls = llm.invoke(messages)

        assert isinstance(
            ai_tool_calls, AIMessage
        ), f"Expected AIMessage got {type(ai_tool_calls)}"
        ai_tool_calls = self._format_response(ai_tool_calls)

        return {
            "messages": [ai_tool_calls],
            "reranker_query": self.get_user_query(state),
        }

    def _get_user_message(self, num_research_iterations: int) -> str:
        if num_research_iterations == 0:
            return PROMPTS["inference/user_new_followup"]
        if num_research_iterations > 2:
            return PROMPTS["inference/user_must_answer"]
        else:
            return PROMPTS["inference/user_retry_after_tools"]

    def _get_messages(self, state: MainState) -> list[BaseMessage]:
        """Strip tool call contents and replace with success message.
        Remove user query message if necessary.
        """
        processed_messages = []
        pattern = r'(<<[^>]*>>).*'

        for message in state.messages:
            if isinstance(message, ToolMessage):
                content = str(message.content)
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    message = ToolMessage(
                        content=match.group(1)
                        + "\n\nTool call successful. Documents merged into context corpus",
                        tool_call_id=message.tool_call_id,
                    )
            processed_messages.append(message)

        # We replace the user query message with a prompt if it is a new turn
        if state.num_research_iterations == 0:
            assert len(state.messages) == state.current_turn_query_index + 1
            processed_messages.pop()

        return processed_messages

    def node_prompt_llm_w_tools(self, state: MainState) -> dict:
        allow_research = state.num_research_iterations <= 2
        system_message = (
            PROMPTS["inference/system_allow_research"]
            if allow_research
            else PROMPTS["inference/system_deny_research"]
        )

        context = "\n\n</ContextDocument>\n\n<ContextDocument>\n\n".join(
            doc.page_content for doc in state.current_context
        )

        messages = [
            ("system", system_message.format(context=context)),
            *self._get_messages(state),
            (
                "human",
                self._get_user_message(state.num_research_iterations).format(
                    user_query=self.get_user_query(state)
                ),
            ),
        ]

        if allow_research:
            # Use the recent LangChain support for combining tools with structured output
            llm = self.llm.bind_tools(
                self.tools, tool_choice="auto", response_format=RAGInferenceOutput
            )
        else:
            llm = self.llm.with_structured_output(
                RAGInferenceOutput, method="json_schema"
            )

        # Create the LCEL chain with RunnableBranch for conditional processing
        chain = llm | {
            STREAM: JsonOutputParser(),
            PASSTHRU: RunnablePassthrough(),
        }

        chunk = None
        for chunk in chain.stream(messages):
            streaming_content = chunk.get(STREAM, {})
            # Stream answer content if it's structured output with answer field i.e. the LLM is not using tools
            if "answer" in streaming_content:
                self._stream_message("token", streaming_content["answer"])

        if chunk is None:
            raise RuntimeError("No response received from LLM chain")

        final_response = self._format_response(chunk[PASSTHRU])
        return {"messages": [final_response]}

    def _tools_condition(self, state: MainState) -> Hashable | list[Hashable]:
        """
        Determine if we should run tools or exit the flow.
        """
        ai_message = state.messages[-1]
        if isinstance(ai_message, AIMessage) and ai_message.tool_calls:
            return [TOOL_NODE, GENERATE_RERANKER_QUERY]
        else:
            return END

    def node_generate_reranker_query(self, state: MainState):
        """Generate a reranker query considering chat history."""
        llm = self.llm.bind(temperature=0)
        messages = [
            ("system", PROMPTS["inference/generate_reranker_query_system"]),
            *state.messages,
        ]
        response = llm.invoke(messages)
        return {"reranker_query": response.content}

    def _get_all_tool_documents(self, state: MainState) -> list[Document]:
        """Extract all Document objects from tool call results in the message history."""
        documents = []

        for message in state.messages:
            if isinstance(message, ToolMessage):
                # The artifact contains the Document objects when using response_format="content_and_artifact"
                if hasattr(message, 'artifact') and message.artifact:
                    if isinstance(message.artifact, list):
                        documents.extend(message.artifact)
                    else:
                        documents.append(message.artifact)

        return documents

    def node_rerank(self, state: MainState):
        reranker = RateLimitedCohereRerank(model="rerank-v3.5", top_n=30)
        reranked_docs = reranker.compress_documents(
            documents=self._get_all_tool_documents(state), query=state.reranker_query
        )

        return {
            "num_research_iterations": state.num_research_iterations + 1,
            "current_context": reranked_docs,
        }

    def _create_tools(self):
        """Create bound tools that don't have self parameters."""

        @tool(response_format="content_and_artifact")
        def semantic_search(
            search_query: Annotated[
                str,
                "The search query to use for semantic search in the vector database. Should be a well-formed query that describes what information you are looking for.",
            ],
        ):
            """The primary data source for information about Peplink products and services.
            Search the vector database for information relevant to the user query by providing
            a semantic search query. This tool is the primary
            means of retrieving information about Peplink products and services but it also
            contains some information about general IT networking concepts that are adjacent
            to Peplink products and services.
            """
            query_embedding = self.pinecone.get_query_embedding(search_query)
            docs = self.pinecone.retrieve(
                search_query, query_embedding, top_k=70, rerank_top_n=30
            )
            content = f"<<Semantic Search: '{search_query}'; Docs: {len(docs)}>>\n\n____FIRST DOC____\n\n{docs[0].page_content}"
            return (content, docs)

        @tool(response_format="content_and_artifact")
        def search_web(
            search_query: Annotated[
                str,
                "A web search for a concept or entity to drill down on",
            ],
        ):
            """Use this tool when you need to drill down on a specific aspect of the user query by performing
            a web search using Google. This tool is most useful for general questions about IT networking. It is not
            useful for specific questions about Peplink products and services. Do not use this tool more than once
            in a given turn.
            """
            response_doc = Document(
                page_content="Example web search results",
                metadata={
                    "source": "web",
                    "search_query": search_query,
                },
            )
            content = f"<<Web: '{search_query}'>>\n\n{response_doc.page_content}"
            return (content, [response_doc])

        @tool(response_format="content_and_artifact")
        def search_wikipedia(
            search_query: Annotated[
                str,
                "The search query to use for Wikipedia search. Should be a specific entity or concept.",
            ],
        ):
            """Use this tool to perform a Wikipedia search when you need general information about a topic that
            is not about Peplink/Pepwave products and services. This can be used to get information specific to
            the IT networking domain or general information from any other domain. It is most useful for
            broad, general concepts that you would typically find in an encyclopedia. Do not use this tool more than
            twice in a given turn.
            """
            response_doc = Document(
                page_content="Example Wikipedia search results",
                metadata={
                    "source": "wikipedia",
                    "search_query": search_query,
                },
            )
            content = f"<<Wikipedia: '{search_query}'>>\n\n{response_doc.page_content}"
            return (content, [response_doc])

        return [semantic_search, search_web, search_wikipedia]

    def compile(self) -> CompiledStateGraph:

        graph_builder = StateGraph(MainState)

        # Nodes
        graph_builder.add_node(ENTRY_POINT, self.node_entry_point)
        graph_builder.add_node(INIT_RETRIEVAL, self.node_init_retrieval)
        graph_builder.add_node(PROMPT_LLM_W_TOOLS, self.node_prompt_llm_w_tools)
        graph_builder.add_node(
            GENERATE_RERANKER_QUERY, self.node_generate_reranker_query
        )
        graph_builder.add_node(
            TOOL_NODE,
            ToolNode(tools=self.tools),
        )
        graph_builder.add_node(RERANK, self.node_rerank)

        # Edges
        graph_builder.add_edge(START, ENTRY_POINT)
        graph_builder.add_conditional_edges(
            ENTRY_POINT,
            self._entry_point_condition,
            [INIT_RETRIEVAL, PROMPT_LLM_W_TOOLS],
        )
        graph_builder.add_edge(INIT_RETRIEVAL, TOOL_NODE)
        graph_builder.add_conditional_edges(
            PROMPT_LLM_W_TOOLS,
            self._tools_condition,
            [TOOL_NODE, GENERATE_RERANKER_QUERY, END],
        )
        graph_builder.add_edge(TOOL_NODE, RERANK)
        graph_builder.add_edge(GENERATE_RERANKER_QUERY, RERANK)
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
            use_responses_api=False,
            # output_version="responses/v1",
        )
