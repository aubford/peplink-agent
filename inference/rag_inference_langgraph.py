from typing import Annotated, Any, cast
from typing_extensions import TypedDict

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.messages import HumanMessage, AIMessage

from inference.history_aware_retrieval_query import (
    get_history_aware_retrieval_query_chain,
)
from inference.cohere_rerank import RateLimitedCohereRerank
from inference.rate_limiters import openai_rate_limiter
from util.root_only_tracer import RootOnlyTracer
from prompts import load_prompts

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages


# Load prompts
PROMPTS = load_prompts()

messages_prompt = ChatPromptTemplate(
    [
        ("system", PROMPTS['inference/system']),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)


# Define the state schema
class RagState(TypedDict):
    """State for RAG inference system."""

    messages: Annotated[list, add_messages]  # Chat history as messages
    query: str  # Current user query
    retrieval_query: str  # Possibly rewritten query based on chat history
    context: list  # Retrieved documents
    answer: str  # Generated answer
    thread_id: str  # Thread identifier for persistence


class RagInferenceLangGraph:
    def __init__(
        self,
        llm_model: str,
        pinecone_index_name: str,
        embedding_model: str = "text-embedding-3-large",
        temperature: float = 1,
        streaming: bool = False,
        eval_llm: BaseChatModel | None = None,
        messages: ChatPromptTemplate = messages_prompt,
        minimal_tracer: bool = False,
    ):
        # Initialize components
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

        self.eval_llm = eval_llm or self.llm
        self.messages = messages
        self.minimal_tracer = minimal_tracer

        # Initialize memory saver for persistence
        self.memory = MemorySaver()

        # Default thread ID for maintaining compatibility with original interface
        self.default_thread_id = "default_conversation"

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Build the RAG graph with nodes for each step in the pipeline."""
        # Create the graph
        graph_builder = StateGraph(RagState)

        # Add nodes for each step in the RAG pipeline
        graph_builder.add_node(
            "generate_retrieval_query", self._generate_retrieval_query
        )
        graph_builder.add_node("retrieve_context", self._retrieve_context)
        graph_builder.add_node("generate_answer", self._generate_answer)
        graph_builder.add_node("update_messages", self._update_messages)

        # Define the graph flow
        graph_builder.add_edge(START, "generate_retrieval_query")
        graph_builder.add_edge("generate_retrieval_query", "retrieve_context")
        graph_builder.add_edge("retrieve_context", "generate_answer")
        graph_builder.add_edge("generate_answer", "update_messages")

        # Compile the graph with memory
        config = {}
        if self.minimal_tracer:
            config["callbacks"] = [RootOnlyTracer(project_name="langchain-pepwave")]

        return graph_builder.compile(checkpointer=self.memory, **config)

    def _generate_retrieval_query(self, state: RagState) -> dict:
        """Generate a retrieval query considering chat history."""
        # Get the history-aware retrieval query
        query_chain = get_history_aware_retrieval_query_chain(llm=self.llm)

        # Convert messages to the format expected by history_aware_retrieval_query
        chat_history = []
        for msg in state["messages"]:
            if msg.get("role") == "user" or isinstance(msg, HumanMessage):
                content = msg.get("content") if isinstance(msg, dict) else msg.content
                chat_history.append(("human", content))
            elif msg.get("role") == "assistant" or isinstance(msg, AIMessage):
                content = msg.get("content") if isinstance(msg, dict) else msg.content
                chat_history.append(("assistant", content))

        retrieval_query = query_chain.invoke(
            {"input": state["query"], "chat_history": chat_history}
        )

        return {"retrieval_query": retrieval_query}

    def _retrieve_context(self, state: RagState) -> dict:
        """Retrieve relevant documents based on the query."""
        context = self.retriever.invoke(state["retrieval_query"])
        return {"context": context}

    def _generate_answer(self, state: RagState) -> dict:
        """Generate an answer based on the context and query."""
        # Convert messages to the format expected by stuff_documents_chain
        chat_history = []
        for msg in state["messages"]:
            if msg.get("role") == "user" or isinstance(msg, HumanMessage):
                content = msg.get("content") if isinstance(msg, dict) else msg.content
                chat_history.append(("human", content))
            elif msg.get("role") == "assistant" or isinstance(msg, AIMessage):
                content = msg.get("content") if isinstance(msg, dict) else msg.content
                chat_history.append(("assistant", content))

        chain = create_stuff_documents_chain(
            self.eval_llm,
            self.messages,
            document_separator="\n\n</ContextDocument>\n\n<ContextDocument>\n\n",
        )

        answer = chain.invoke(
            {
                "input": state["query"],
                "chat_history": chat_history,
                "context": state["context"],
            }
        )

        return {"answer": answer}

    def _update_messages(self, state: RagState) -> dict:
        """Update the message history with the new query and answer."""
        # Use LangGraph's add_messages annotator to manage message state
        return {
            "messages": [
                {"role": "user", "content": state["query"]},
                {"role": "assistant", "content": state["answer"]},
            ]
        }

    def query(self, query: str) -> dict:
        """
        Process a single query through the RAG pipeline.
        This maintains the identical interface to the original RagInference.

        Args:
            query: The user's question

        Returns:
            dict: Contains the answer and other chain outputs
        """
        # Use the default thread ID for maintaining conversation state
        config = {"configurable": {"thread_id": self.default_thread_id}}

        # Initialize state
        initial_state = {"query": query, "thread_id": self.default_thread_id}

        # Invoke the graph
        result = cast(Any, self.graph).invoke(initial_state, config=config)

        # Update instance chat_history for compatibility with original interface
        self.chat_history = self._convert_messages_to_tuples(result.get("messages", []))

        # Return the result with answer for compatibility
        return {
            "answer": result.get("answer", ""),
            "context": result.get("context", []),
            "retrieval_query": result.get("retrieval_query", ""),
        }

    def clear_history(self) -> None:
        """Clear the chat history."""
        # Clear the instance variable for backward compatibility
        self.chat_history = []

        # Set empty state for the default thread
        try:
            cast(Any, self.memory).clear(thread_id=self.default_thread_id)
        except:
            # Fallback if clear method is not available
            empty_state = {
                "messages": [],
                "query": "",
                "retrieval_query": "",
                "context": [],
                "answer": "",
                "thread_id": self.default_thread_id,
            }
            config = {"configurable": {"thread_id": self.default_thread_id}}
            cast(Any, self.graph).invoke(empty_state, config=config)

    async def batch_query_for_eval(
        self, queries: dict[str, str]
    ) -> dict[str, dict[str, Any]]:
        """
        Process multiple queries in parallel for evaluation purposes.

        Args:
            queries: Dictionary of query identifiers to queries

        Returns:
            dict: Mapping of query identifiers to results
        """
        results = {}

        # Process each query individually (no chat history)
        for query_id, query in queries.items():
            # Create a new thread_id for each query
            thread_id = f"eval_{query_id}"

            # Initialize state for fresh conversation
            initial_state = {
                "messages": [],
                "query": query,
                "retrieval_query": "",
                "context": [],
                "answer": "",
                "thread_id": thread_id,
            }

            # Add thread_id to config
            config = {"configurable": {"thread_id": thread_id}}

            # Process query
            result = self.graph.invoke(initial_state, config=config)

            # Format result to match original implementation
            formatted_result = {
                "query_id": query_id,
                "answer": result.get("answer", ""),
                "custom_id": result.get("answer", ""),
                "context": result.get("context", []),
                "retrieval_query": result.get("retrieval_query", ""),
            }
            results[query_id] = formatted_result

            # Clean up the thread to avoid memory leaks
            try:
                cast(Any, self.memory).clear(thread_id=thread_id)
            except:
                pass

        return results

    def _convert_messages_to_tuples(self, messages: list) -> list[tuple[str, str]]:
        """Convert LangGraph messages to the tuple format used by the original implementation."""
        result = []
        for msg in messages:
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    result.append(("human", msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    result.append(("assistant", msg.get("content", "")))
            elif isinstance(msg, HumanMessage):
                result.append(("human", msg.content))
            elif isinstance(msg, AIMessage):
                result.append(("assistant", msg.content))
        return result
