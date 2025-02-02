from __future__ import annotations

from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.retrievers import RetrieverLike, RetrieverOutputLike
from langchain_core.runnables import RunnableBranch
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


def history_aware_retrieval_query() -> str:
    """Given a chat history, summarize it into a single question."""

    contextualize_q_system_prompt = (
        "Given the following chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    if "input" not in prompt.input_variables:
        raise ValueError("Expected `input` to be a prompt variable, " f"but got {prompt.input_variables}")

    retrieve_documents: str = RunnableBranch(
        (
            # Both empty string and empty list evaluate to False
            lambda x: not x.get("chat_history", False),
            # If no chat history, then we just pass input to retriever
            (lambda x: x["input"]),
        ),
        # If chat history, then we pass inputs to LLM chain, then to retriever
        prompt | ChatOpenAI(model="o1-mini") | StrOutputParser(),
    ).with_config(run_name="chat_retriever_history_aware_query")
    return retrieve_documents
