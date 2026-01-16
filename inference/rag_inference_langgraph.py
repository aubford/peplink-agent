import re
import os
from typing import Annotated, Hashable, Literal, TypedDict, TypeGuard
from langchain_core.documents import Document
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.tools import ToolException, tool
from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    ToolMessage,
)
from langgraph.constants import END, START
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.config import get_stream_writer
from inference.cohere_rerank import RateLimitedCohereRerank
from inference.rag_inference import InferenceBase
from inference.pinecone_retriever import PineconeRetriever
from inference.rate_limiters import openai_rate_limiter
from prompts import load_prompts
from util.document_utils import documents_to_dicts, dict_to_document, get_docs_of_type


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
PROMPT_LLM_MAIN = "NODE__PROMPT_LLM_MAIN"
PROMPT_LLM_DEMAND_ANSWER = "NODE__PROMPT_LLM_DEMAND_ANSWER"
GENERATE_RERANKER_QUERY = "NODE__GENERATE_RERANKER_QUERY"
TOOL_NODE = "NODE__TOOL_NODE"
HANDLE_TOOL_RESULTS = "NODE__HANDLE_TOOL_RESULTS"
RERANK = "NODE__RERANK"
STREAM = "STREAM"
PASSTHRU = "PASSTHRU"


class StreamMessage(BaseModel):
    type: Literal["token", "messages", "complete", "error", "log"]
    content: str


# @tool
# def decide_have_enough_info(
#     have_enough_information: Annotated[
#         bool,
#         "True if the context documents are sufficient to answer the user's question accurately and confidently.",
#     ],
#     have_enough_information_reasoning: Annotated[
#         str,
#         "Reasoning behind whether there is enough information in the current context.",
#     ],
#     answer: Annotated[
#         str,
#         "The answer to the user query if there is enough information; otherwise, return an empty string.",
#     ],
# ):
#     """Decide whether there's enough information in the current context to answer the user query accurately and explain the reasoning. If there is enough, provide an answer."""
#     raise ValueError("decide_have_enough_info function body should never be called")


class RAGInferenceOutput(BaseModel):
    have_enough_information: bool = Field(
        description="True if the context documents are sufficient to answer the user's question accurately and confidently."
    )
    have_enough_information_reasoning: str = Field(
        description="Reasoning behind whether there is enough information in the current context."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification. Leave blank if there is enough information in the current context."
    )
    answer: str = Field(
        description="The answer to the user query; if there is not enough information in the context to answer with confidence, return an empty string. Do not refer to the context corpus in your answer, simply answer the question directly."
    )


# Define the state schema using Pydantic
class MainState(BaseModel):
    """State for RAG inference system using Pydantic.

    Attributes:
        current_context: the actual docs exposed to the LLM in system msg
        messages: What the user sees
        current_turn_query_index: the index of the user query message in the messages list for the current turn
        num_research_iterations: number of times research has been done for a single user query
        reranker_query: query for the reranker
        reranker_query_for_query_index: what user query the current reranker query is based on i.e. does it need to catch up to the current one?
        thread_id: the thread id for the user query
    """

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    num_research_iterations: int = 0
    current_turn_query_index: int = 0
    reranker_query_for_query_index: int = 0
    current_context: list[Document] = Field(default_factory=list)
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
        self.wikipedia_retriever = WikipediaRetriever(load_max_docs=3)
        self.checkpointer = checkpointer or InMemorySaver()
        self.research_tools = self._create_research_tools()

    # todo: remove
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
            return PROMPT_LLM_MAIN
        else:
            return INIT_RETRIEVAL

    def _format_will_do_research_response_ai_msg(
        self,
        ai_message: AIMessage,
        intro_text: str = "I'll research your question using these tools:",
    ) -> AIMessage:
        """Format the response from the LLM whether it be a tool call or the final answer."""
        if not ai_message.tool_calls:
            raise ValueError(
                "No tool calls were made; init retrieval should only return tool calls."
            )

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
            ("system", PROMPTS["inference/retrieval_system"].format(addendum="")),
            (
                "human",
                PROMPTS["inference/init_retrieval"].format(
                    user_query=self.get_user_query(state)
                ),
            ),
        ]
        llm = self.llm.bind_tools(self.research_tools, tool_choice="required")
        ai_msg_w_tool_calls = llm.invoke(messages)

        assert isinstance(
            ai_msg_w_tool_calls, AIMessage
        ), f"Expected AIMessage got {type(ai_msg_w_tool_calls)}"
        ai_msg_w_tool_calls = self._format_will_do_research_response_ai_msg(
            ai_msg_w_tool_calls
        )

        return {
            "messages": [ai_msg_w_tool_calls],
            "reranker_query": self.get_user_query(state),
        }

    @staticmethod
    def _format_context_documents(documents: list[Document]) -> str:
        """Join document page contents with ContextDocument separator."""
        return "\n\n</ContextDocument>\n\n<ContextDocument>\n\n".join(
            doc.page_content for doc in documents
        )

    def _replace_tool_call_msgs_w_success_msg(
        self, messages: list[BaseMessage]
    ) -> list[BaseMessage]:
        """Strip tool call contents and replace with success message so we aren't sending context documents to the model twice."""
        processed_messages = []
        pattern = r'(<<[^>]*>>).*'

        for message in messages:
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
        return processed_messages

    @staticmethod
    def _convert_answer_tool_call_to_message(res: AIMessage) -> AIMessage:
        answer = res.tool_calls[0]["args"]["answer"]
        return AIMessage(content=answer)

    def node_prompt_llm_main(self, state: MainState) -> dict:
        """Determine if we need to do further research and then either provide an answer or make research tool calls

        state.messages: [system, human, AI(will do research), *tool calls(with artifacts and first docs)] OR [system, Human, AI(will do research), *tool calls(with artifacts and first docs), AI(answer), Human]
        """
        system_message = PROMPTS["inference/system_allow_research_decide"]
        context = self._format_context_documents(state.current_context)

        messages = [
            ("system", system_message.format(context=context)),
            *self._replace_tool_call_msgs_w_success_msg(state.messages),
        ]

        decide_llm = self.llm.with_structured_output(RAGInferenceOutput)
        decide_res = decide_llm.invoke(messages)
        assert isinstance(
            decide_res, RAGInferenceOutput
        ), f"Expected RAGInferenceOutput got {type(decide_res)}"
        if decide_res.have_enough_information:
            assert decide_res.answer, "Answer cannot be empty"
            return {"messages": [AIMessage(content=decide_res.answer)]}

        # Not enough information: Make research tool calls
        addendum = "- Reflect on search query tool calls that have been made previously in the conversation and avoid repeating the same search queries."
        research_messages = [
            ("system", PROMPTS["inference/retrieval_system"].format(addendum=addendum)),
            *self._replace_tool_call_msgs_w_success_msg(state.messages),
        ]

        is_new_user_query = state.num_research_iterations == 0
        if is_new_user_query:
            research_messages.pop()

        research_messages.append(
            HumanMessage(
                content=PROMPTS["inference/user_knowledge_gap"].format(
                    knowledge_gap=decide_res.knowledge_gap,
                    user_query=self.get_user_query(state),
                )
            )
        )

        llm = self.llm.bind_tools(self.research_tools, tool_choice="required")
        ai_msg_w_tool_calls = llm.invoke(research_messages)

        assert isinstance(
            ai_msg_w_tool_calls, AIMessage
        ), f"Expected AIMessage got {type(ai_msg_w_tool_calls)}"
        assert len(ai_msg_w_tool_calls.tool_calls) > 1, "Expected multiple tool calls"

        ai_msg_w_tool_calls = self._format_will_do_research_response_ai_msg(
            ai_msg_w_tool_calls
        )

        return {"messages": [ai_msg_w_tool_calls]}

    def _tools_condition(self, state: MainState) -> Hashable | list[Hashable]:
        """
        Determine if we should run tools or exit the flow.
        """
        ai_message = state.messages[-1]
        if isinstance(ai_message, AIMessage) and ai_message.tool_calls:
            return [TOOL_NODE, GENERATE_RERANKER_QUERY]
        else:
            return END

    def _allow_research_condition(self, state: MainState) -> Hashable | list[Hashable]:
        allow_research = state.num_research_iterations <= 2
        if allow_research:
            return PROMPT_LLM_MAIN
        else:
            return PROMPT_LLM_DEMAND_ANSWER

    def node_prompt_llm_demand_answer(self, state: MainState) -> dict:
        context = self._format_context_documents(state.current_context)
        system_message = PROMPTS["inference/system_deny_research"]
        llm = self.llm
        messages = [
            ("system", system_message.format(context=context)),
            *self._replace_tool_call_msgs_w_success_msg(state.messages),
        ]
        response = llm.invoke(messages)
        return {
            "messages": [response],
        }

    def node_generate_reranker_query(self, state: MainState):
        """Generate a reranker query considering chat history."""
        if state.reranker_query_for_query_index == state.current_turn_query_index:
            return

        llm = self.llm
        # Verify that current_turn_query_index points to a HumanMessage
        if state.current_turn_query_index >= len(state.messages):
            raise ValueError(
                f"current_turn_query_index {state.current_turn_query_index} is out of bounds"
            )

        current_message = state.messages[state.current_turn_query_index]
        if not isinstance(current_message, HumanMessage):
            raise ValueError(
                f"current_turn_query_index {state.current_turn_query_index} does not point to a HumanMessage"
            )

        # Get all messages up to and including the current user query
        reranker_messages = state.messages[: state.current_turn_query_index + 1]
        messages = [
            ("system", PROMPTS["inference/generate_reranker_query_system"]),
            *self._replace_tool_call_msgs_w_success_msg(reranker_messages),
        ]
        response = llm.invoke(messages)
        return {
            "reranker_query": response.content,
            "reranker_query_for_query_index": state.current_turn_query_index,
        }

    @staticmethod
    def _is_not_duplicate_doc_dict(doc: dict, docs: list[Document]):
        existing_ids = [d.metadata["id"] for d in docs if "id" in d]
        doc_id = doc["metadata"].get("id", None)
        return doc_id not in existing_ids

    def _get_all_tool_artifact_documents(self, state: MainState) -> list[Document]:
        """Extract all Document objects from ToolMessage artifact lists in the message history.

        Artifacts are stored as dicts in state (for serialization), so we convert them
        back to Document objects here.
        """
        documents = []

        for message in state.messages:
            if isinstance(message, ToolMessage):
                # The artifact contains dict representations of Document objects when using response_format="content_and_artifact"
                assert (
                    hasattr(message, 'artifact') and message.artifact is not None
                ), "ToolMessage must have an artifact"
                assert isinstance(
                    message.artifact, list
                ), f"Expected artifact to be a list, got {type(message.artifact)}"

                for doc in message.artifact:
                    assert isinstance(
                        doc, dict
                    ), f"Expected artifact list items to be dicts, got {type(doc)}"
                    # dedupe
                    if self._is_not_duplicate_doc_dict(doc, documents):
                        documents.append(dict_to_document(doc))

        return documents

    @staticmethod
    def _select_top_docs(
        reranked_docs: list[Document],
        top_n: int = 30,
        min_wikipedia_docs: int = 2,
    ) -> list[Document]:
        top_docs = list(reranked_docs[:top_n])
        other_docs = list(reranked_docs[top_n:])
        wikipedia_in_other = get_docs_of_type(other_docs, "wikipedia")
        wikipedia_in_top = get_docs_of_type(top_docs, "wikipedia")
        min_wikipedia_docs = min(len(wikipedia_in_other), min_wikipedia_docs)
        needed_wikipedia = min_wikipedia_docs - len(wikipedia_in_top)
        if needed_wikipedia < 1:
            return top_docs
        replacements = wikipedia_in_other[:needed_wikipedia]
        top_docs[-len(replacements) :] = replacements
        return top_docs

    def node_rerank(self, state: MainState):
        """Get all the documents that have been fetched so far and select the top 30 that are applicable to the current conversation"""
        all_docs = self._get_all_tool_artifact_documents(state)
        reranker = RateLimitedCohereRerank(model="rerank-v3.5", top_n=None)
        reranked_docs = reranker.compress_documents(
            documents=all_docs,
            query=state.reranker_query,
        )
        top_docs = self._select_top_docs(reranked_docs)

        return {
            "num_research_iterations": state.num_research_iterations + 1,
            "current_context": top_docs,
        }

    def _create_research_tools(self):
        """Create bound tools that don't have self parameters."""

        @tool(response_format="content_and_artifact")
        def semantic_search(
            search_query: Annotated[
                str,
                "The semantic search query. It should be a well-formed query that describes what information you are looking for and is optimized for semantic search in a vector database",
            ],
        ):
            """Use this tool when you need information about Peplink or Pepwave cellular networking products and services or tangential IT networking topics. This is the primary data source and should be used more than the other tools. Use this tool 1-4 times in a given turn. Avoid repeating previously searched queries."""
            print(f"START: semantic_search for search_query: {search_query}")
            query_embedding = self.pinecone.get_query_embedding(search_query)
            # reranking evaluates query<>document directly and is more accurate than vectore similarity; cast a wide net with top_k and then rerank to get the best matches
            docs = self.pinecone.retrieve(
                search_query, query_embedding, top_k=70, rerank_top_n=30
            )
            content = f"<<Semantic Search: '{search_query}'; Docs: {len(docs)}>>\n\n____FIRST DOC____\n\n{docs[0].page_content}"
            # Convert Documents to dicts for state serialization
            print(f"END: semantic_search for search_query: {search_query}")
            return content, documents_to_dicts(docs)

        @tool(response_format="content_and_artifact")
        def search_web(
            search_query: Annotated[
                str,
                "A web search for a concept or entity to drill down on",
            ],
        ):
            """Use this tool to drill down on a specific aspect of the user query by performing a web search using Tavily. This tool is most useful for general questions about IT networking. Do not use to research Peplink or Pepwave cellular networking products and services. Do not Use this tool 0-2 times in a given turn. Avoid repeating previously searched queries."""
            tavily_api_key = os.environ.get("TAVILY_API_KEY")
            if not tavily_api_key:
                content = f"<<Web: '{search_query}'; Docs: 0>>\n\nMissing TAVILY_API_KEY in .env."
                return content, []
            tavily = TavilySearch(
                tavily_api_key=tavily_api_key,
                max_results=5,
                include_answer=False,
                include_raw_content=False,
            )

            try:
                results = tavily.invoke({"query": search_query})
            except ToolException as exc:
                content = f"<<Web: '{search_query}'; Docs: 0>>\n\n{exc}"
                return content, []

            if isinstance(results, dict) and results.get("error"):
                content = f"<<Web: '{search_query}'; Docs: 0>>\n\n{results['error']}"
                return content, []

            docs = []
            for result in results.get("results", []):
                docs.append(
                    Document(
                        page_content=result.get("content", ""),
                        metadata={
                            "type": "web",
                            "search_query": search_query,
                            "title": result.get("title"),
                            "url": result.get("url"),
                            "score": result.get("score"),
                        },
                    )
                )

            if docs:
                content = f"<<Web: '{search_query}'; Docs: {len(docs)}>>\n\n____FIRST DOC____\n\n{docs[0].page_content}"
            else:
                content = f"<<Web: '{search_query}'; Docs: 0>>\n\nNo results found."
            return content, documents_to_dicts(docs)

        @tool(response_format="content_and_artifact")
        def search_wikipedia(
            search_query: Annotated[
                str,
                "The search query to use for Wikipedia search. Should be a specific entity or concept.",
            ],
        ):
            """Use this tool to perform a Wikipedia search when you need general information about a specific term or topic mentioned in the user query. Do not use to research Peplink or Pepwave cellular networking products and services. This can be used to get information specific to the IT networking domain or general information from any other domain other than Peplink or Pepwave products and services. It is most useful for researching broad, general topics that you would typically find in an encyclopedia or to lookup the definition of a specific term. Use this tool 0-2 times in a given turn. Avoid repeating previously searched queries."""
            docs = self.wikipedia_retriever.invoke(search_query)
            if docs:
                content = f"<<Wikipedia: '{search_query}'; Docs: {len(docs)}>>\n\n____FIRST DOC____\n\n{docs[0].page_content}"
                return content, []

            content = f"<<Wikipedia: '{search_query}'; Docs: 0>>\n\nNo results found."
            for doc in docs:
                doc.metadata["type"] = "wikipedia"
            return content, documents_to_dicts(docs)

        return [
            semantic_search,
            search_web,
            search_wikipedia,
        ]

    def compile(self) -> CompiledStateGraph:

        graph_builder = StateGraph(MainState)

        # Nodes
        graph_builder.add_node(ENTRY_POINT, self.node_entry_point)
        graph_builder.add_node(INIT_RETRIEVAL, self.node_init_retrieval)
        graph_builder.add_node(PROMPT_LLM_MAIN, self.node_prompt_llm_main)
        graph_builder.add_node(
            GENERATE_RERANKER_QUERY, self.node_generate_reranker_query
        )
        graph_builder.add_node(
            TOOL_NODE,
            ToolNode(tools=self.research_tools),
        )
        graph_builder.add_node(RERANK, self.node_rerank)
        graph_builder.add_node(
            PROMPT_LLM_DEMAND_ANSWER, self.node_prompt_llm_demand_answer
        )

        # Edges
        graph_builder.add_edge(START, ENTRY_POINT)
        graph_builder.add_conditional_edges(
            ENTRY_POINT,
            self._entry_point_condition,
            [INIT_RETRIEVAL, PROMPT_LLM_MAIN],
        )
        graph_builder.add_edge(INIT_RETRIEVAL, TOOL_NODE)
        graph_builder.add_conditional_edges(
            PROMPT_LLM_MAIN,
            self._tools_condition,
            [TOOL_NODE, GENERATE_RERANKER_QUERY, END],
        )
        graph_builder.add_edge(TOOL_NODE, RERANK)
        graph_builder.add_edge(GENERATE_RERANKER_QUERY, RERANK)
        graph_builder.add_conditional_edges(
            RERANK,
            self._allow_research_condition,
            [PROMPT_LLM_MAIN, PROMPT_LLM_DEMAND_ANSWER],
        )
        graph_builder.add_edge(PROMPT_LLM_DEMAND_ANSWER, END)

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
        if os.getenv("DEVELOPMENT") == "true":
            return ChatOpenAI(
                model=self.llm_model,
                temperature=self.temperature,
                # todo: revert
                streaming=False,
                reasoning_effort="low",
                rate_limiter=openai_rate_limiter,
                stream_usage=True,
                use_responses_api=False,
                http_client=self.client_with_logging,
                http_async_client=self.async_client_with_logging,
                # output_version="responses/v1",
            )
        else:
            return ChatOpenAI(
                model=self.llm_model,
                temperature=self.temperature,
                streaming=self.streaming,
                service_tier="priority",
                reasoning_effort="low",
                rate_limiter=openai_rate_limiter,
                use_responses_api=False,
            )
