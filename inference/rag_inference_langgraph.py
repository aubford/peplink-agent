from typing import Annotated, Any, Optional, cast
from typing_extensions import TypedDict

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

from inference.history_aware_retrieval_query import (
    get_history_aware_retrieval_query_chain,
)
from inference.cohere_rerank import RateLimitedCohereRerank
from inference.rate_limiters import openai_rate_limiter
from util.root_only_tracer import RootOnlyTracer
from prompts import load_prompts

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph


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
        self.memory = InMemorySaver()

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
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
        query_chain = get_history_aware_retrieval_query_chain(llm=self.llm)

        # Pass messages as-is (list of dicts)
        chat_history = state["messages"]

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
        # Pass messages as-is (list of dicts)
        chat_history = state["messages"]

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
        return {
            "messages": [
                {"role": "user", "content": state["query"]},
                {"role": "assistant", "content": state["answer"]},
            ]
        }

    def query(self, query: str, thread_id: Optional[str] = None) -> dict:
        """
        Process a query through the RAG pipeline.

        Args:
            query: The user's question
            thread_id: Optional thread identifier for conversation persistence

        Returns:
            dict: The final state containing the answer and other outputs
        """
        # Create config with thread_id if provided
        config = {"configurable": {"thread_id": thread_id}} if thread_id else {}

        # Determine initial state based on whether we have a thread_id
        if thread_id:
            # If we have a thread_id, we're continuing a conversation
            # Initialize with just the new query
            initial_state = {"query": query, "thread_id": thread_id}
        else:
            # Starting a new conversation
            initial_state = {
                "messages": [],
                "query": query,
                "retrieval_query": "",
                "context": [],
                "answer": "",
                "thread_id": "",
            }

        # Invoke the graph
        result = self.graph.invoke(initial_state, config=config)
        return result

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
            # Create a new thread_id for each query to isolate them
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

            # Add to results with the same format as the original
            result_with_id = {
                **result,
                "query_id": query_id,
                "custom_id": result.get("answer", ""),
            }
            results[query_id] = result_with_id

            # Clean up the thread by setting an empty state
            empty_state = {
                "messages": [],
                "query": "",
                "retrieval_query": "",
                "context": [],
                "answer": "",
                "thread_id": thread_id,
            }
            self.graph.invoke(empty_state, config=config)

        return results
