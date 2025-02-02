from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


def history_aware_retrieval_query() -> RunnableBranch:
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

    return RunnableBranch(
        (
            # Both empty string and empty list evaluate to False
            lambda x: not x.get("chat_history", False),
            # If no chat history, then we just pass input to retriever
            (lambda x: x["input"]),
        ),
        # If chat history, then we pass inputs to LLM chain, then to retriever
        prompt | ChatOpenAI(model_name="o1-mini") | StrOutputParser(),
    ).with_config(run_name="history_aware_retrieval_query")
