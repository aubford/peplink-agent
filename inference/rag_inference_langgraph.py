from typing import Annotated
from typing_extensions import TypedDict

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.runnables.graph import MermaidDrawMethod

from inference.history_aware_retrieval_query import (
    get_history_aware_retrieval_query_chain,
)
from inference.cohere_rerank import RateLimitedCohereRerank
from inference.rag_inference import InferenceBase
from prompts import load_prompts

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from load.batch_manager import BatchManager
from evals.batch_llm import BatchChatOpenAI


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


class RagInferenceLangGraph(InferenceBase):
    def __init__(
        self,
        llm_model: str,
        pinecone_index_name: str,
        embedding_model: str = "text-embedding-3-large",
        temperature: float = 1,
        streaming: bool = False,
        minimal_tracer: bool = False,
    ):
        super().__init__(
            llm_model=llm_model,
            pinecone_index_name=pinecone_index_name,
            embedding_model=embedding_model,
            temperature=temperature,
            streaming=streaming,
            minimal_tracer=minimal_tracer
        )

        self.output_llm = self.llm
        self.conversation_template = None

        # Initialize memory saver for persistence
        self.memory = InMemorySaver()

    def compile(
        self,
        conversation_template: BasePromptTemplate,
        batch_manager: BatchManager | None = None,
    ) -> CompiledStateGraph:
        self.conversation_template = conversation_template
        if batch_manager is not None:
            self.output_llm = BatchChatOpenAI(
                model=self.llm_model,
                batch_manager=batch_manager,
            )

        graph_builder = StateGraph(RagState)

        # Nodes
        graph_builder.add_node(
            "generate_retrieval_query", self._generate_retrieval_query
        )
        graph_builder.add_node("retrieve_context", self._retrieve_context)
        graph_builder.add_node("generate_answer", self._generate_answer)
        graph_builder.add_node("update_messages", self._update_messages)

        # Edges
        graph_builder.add_edge(START, "generate_retrieval_query")
        graph_builder.add_edge("generate_retrieval_query", "retrieve_context")
        graph_builder.add_edge("retrieve_context", "generate_answer")
        graph_builder.add_edge("generate_answer", "update_messages")

        # Compile
        compiled_graph = graph_builder.compile(checkpointer=self.memory)
        self._draw_graph(compiled_graph)
        graph = compiled_graph.with_config(self.config)
        return graph

    def _generate_retrieval_query(self, state: RagState) -> dict:
        print(state)
        """Generate a retrieval query considering chat history."""
        query_chain = get_history_aware_retrieval_query_chain(llm=self.llm)

        retrieval_query = query_chain.invoke(
            {"input": state["query"], "chat_history": state["messages"]}
        )

        return {"retrieval_query": retrieval_query}

    def _retrieve_context(self, state: RagState) -> dict:
        """Retrieve relevant documents based on the query."""
        retriever_base = self.vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 30, "fetch_k": 50}
        )
        compressor = RateLimitedCohereRerank(model="rerank-v3.5", top_n=20)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever_base
        )
        context = retriever.invoke(state["retrieval_query"])
        return {"context": context}

    def _generate_answer(self, state: RagState) -> dict:
        """Generate an answer based on the context and query."""
        assert self.conversation_template
        chain = create_stuff_documents_chain(
            self.output_llm,
            self.conversation_template,
            document_separator="\n\n</ContextDocument>\n\n<ContextDocument>\n\n",
        )

        answer = chain.invoke(
            {
                "input": state["query"],
                "chat_history": state["messages"],
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

    def _draw_graph(self, compiled_graph: CompiledStateGraph):
        compiled_graph.get_graph().print_ascii()
        compiled_graph.get_graph(xray=True).draw_mermaid_png(
            output_file_path="graph_diagram.png",
            draw_method=MermaidDrawMethod.PYPPETEER,
            max_retries=3,
            retry_delay=2.0,
        )
