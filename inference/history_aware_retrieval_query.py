# %%
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, Runnable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain_core.messages.utils import convert_to_messages
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
    ChatMessage,
)
import textwrap
from typing import Any, Dict

prompt = (
    """## INSTRUCTIONS:

"""
    "Given the following chat history and user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
    + textwrap.dedent(
        """

    ## Chat history to use as context:

    {chat_history}

    ## User question to reformulate to include referenced context from the chat history:

    {query}
    """
    )
)


def normalize_messages(
    messages: list[tuple[str, str]],
    human_prefix: str = "User",
    ai_prefix: str = "Assistant",
) -> list[tuple[str, str]]:
    chat_messages = []
    for m in convert_to_messages(messages):
        if isinstance(m, HumanMessage):
            chat_messages.append((human_prefix, m.content))
        elif isinstance(m, AIMessage):
            chat_messages.append((ai_prefix, m.content))
        elif isinstance(m, SystemMessage):
            chat_messages.append(("System", m.content))
        elif isinstance(m, FunctionMessage):
            continue
        elif isinstance(m, ToolMessage):
            continue
        elif isinstance(m, ChatMessage):
            chat_messages.append((m.role, m.content))
        else:
            msg = f"Got unsupported message type: {m}"
            raise ValueError(msg)  # noqa: TRY004
    return chat_messages


def format_messages(
    messages: list[tuple[str, str]],
    human_prefix: str = "User",
    ai_prefix: str = "Assistant",
) -> str:
    return "\n\n".join(
        [
            f"""<{role}>{message}</{role}>"""
            for role, message in normalize_messages(messages, human_prefix, ai_prefix)
        ]
    )


prompt_chain = {
    "chat_history": lambda x: format_messages(x["chat_history"]),
    "query": lambda x: x["query"],
} | PromptTemplate.from_template(prompt)


def _has_no_chat_history(state: Dict[str, Any]) -> bool:
    """Check if chat history is empty and print the check for debugging."""
    chat_history = state.get("chat_history")
    # Both empty string and empty list evaluate to False
    return not chat_history


def get_history_aware_retrieval_query_chain(llm: BaseChatModel) -> Runnable:
    """Given a chat history, summarize it into a single question."""

    return RunnableBranch(
        (
            _has_no_chat_history,
            # If no chat history, then we just pass input to retriever
            (lambda x: x["query"]),
        ),
        # If chat history, then we pass inputs to LLM chain, then to retriever
        prompt_chain | llm | StrOutputParser(),
    ).with_config(run_name="history_aware_retrieval_query")
