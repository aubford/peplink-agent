from typing import Annotated

from langchain_core.documents import Document

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import BasePromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.runnables.graph import MermaidDrawMethod

from inference.history_aware_retrieval_query import (
    get_history_aware_retrieval_query_chain,
)
from inference.cohere_rerank import RateLimitedCohereRerank
from inference.rag_inference import InferenceBase
from inference.pinecone_retriever import PineconeRetriever
from prompts import load_prompts

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from load.batch_manager import BatchManager
from evals.batch_llm import BatchChatOpenAI
from pydantic import BaseModel, Field


# Load prompts
PROMPTS = load_prompts()


# Define the state schema using Pydantic
class RagState(BaseModel):
    """State for RAG inference system using Pydantic."""

    messages: Annotated[list, add_messages] = Field(default_factory=list)
    query: str = ""
    retrieval_query: str = ""
    retrieval_query_embedding: list[float] = Field(default_factory=list)
    context: list = Field(default_factory=list)
    context_history: list = Field(default_factory=list)
    cached_extra_context: list = Field(default_factory=list)
    answer: str = ""
    thread_id: str = "default"
    cached_web_search: str | None = None
    tool_call_count: int = 0


class RagInferenceLangGraph(InferenceBase):
    def __init__(
        self,
        llm_model: str,
        pinecone_index_name: str,
        use_cohere: bool = False,
        **kwargs,
    ):
        super().__init__(
            llm_model=llm_model, pinecone_index_name=pinecone_index_name, **kwargs
        )

        self.output_llm = self.llm
        self.conversation_template = None
        self.use_cohere = use_cohere

        # Initialize the Pinecone retriever
        self.pinecone = PineconeRetriever(
            index_name=pinecone_index_name, embedding_model=self.embedding_model
        )

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

    def _get_cohere_retriever(self) -> ContextualCompressionRetriever:
        retriever_base = self.vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 60, "fetch_k": 100}
        )
        compressor = RateLimitedCohereRerank(model="rerank-v3.5", top_n=40)
        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever_base
        )

    def _generate_retrieval_query(self, state: RagState) -> dict:
        """Generate a retrieval query considering chat history."""
        retrieval_query_chain = get_history_aware_retrieval_query_chain(llm=self.llm)

        retrieval_query = retrieval_query_chain.invoke(
            {"query": state.query, "chat_history": state.messages}
        )

        if self.use_cohere:
            return {"retrieval_query": retrieval_query}

        retrieval_query_embedding = self.pinecone.get_query_embedding(retrieval_query)
        return {
            "retrieval_query": retrieval_query,
            "retrieval_query_embedding": retrieval_query_embedding,
        }

    def _retrieve_context(self, state: RagState) -> dict:
        """Retrieve relevant documents based on the query."""
        if self.use_cohere:
            retriever = self._get_cohere_retriever()
            retrieved_context = retriever.invoke(state.retrieval_query)
        else:
            assert state.retrieval_query_embedding
            retrieved_context = self.pinecone.retrieve(
                state.retrieval_query, state.retrieval_query_embedding
            )

        return {
            "context": retrieved_context[0:20],
            "cached_extra_context": retrieved_context[20:],
            # todo: summary history in background thread
            "context_history": [
                *state.context_history,
                *state.context,
            ],
        }

    def _generate_answer(self, state: RagState) -> dict:
        """Generate an answer based on the context and query."""
        assert self.conversation_template

        messages = self.conversation_template.invoke(
            {
                "query": state.query,
                "chat_history": state.messages,
                "context": "\n\n</ContextDocument>\n\n<ContextDocument>\n\n".join(
                    [doc.page_content for doc in state.context]
                ),
            }
        )
        answer = self.output_llm.invoke(messages)
        return {"answer": answer.content}

    def _update_messages(self, state: RagState) -> dict:
        """Update the message history with the new query and answer."""
        return {
            "messages": [
                {"role": "user", "content": state.query},
                {"role": "assistant", "content": state.answer},
            ]
        }

    def _draw_graph(self, compiled_graph: CompiledStateGraph):
        # compiled_graph.get_graph().print_ascii()
        compiled_graph.get_graph(xray=True).draw_mermaid_png(
            output_file_path="graph_diagram.png",
            draw_method=MermaidDrawMethod.PYPPETEER,
            max_retries=3,
            retry_delay=2.0,
        )
