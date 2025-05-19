import json
import hashlib
from typing import Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from load.batch_manager import BatchManager


class BatchChatOpenAI(BaseChatModel):
    """A chat model that writes to OpenAI batch files instead of making API calls.

    This mock implementation redirects calls to the OpenAI API to a batch file
    that can be processed later using OpenAI's Batch API.
    """

    model_name: str = Field(default="gpt-4.1-nano", alias="model")
    """Model name to use."""
    temperature: float = 0.0
    """What sampling temperature to use."""
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    system_prompt: str = "You are a helpful assistant."
    """Default system prompt to use if none is provided in messages."""
    batch_manager: BatchManager

    @staticmethod
    def hash_messages(messages: str) -> str:
        """Create a deterministic hash of the message content for tracking."""
        # Serialize messages to a canonical string
        return hashlib.md5(messages.encode("utf-8")).hexdigest()

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Write the request to a batch file instead of calling the API."""
        system_prompt = self.system_prompt
        human_messages_for_hash = ""
        formatted_messages = []
        for message in messages:
            if message.type == "system":
                system_prompt = str(message.content)
            elif message.type == "human":
                human_messages_for_hash += str(message.content)
                formatted_messages.append(
                    {"role": "user", "content": str(message.content)}
                )
            elif message.type == "ai":
                formatted_messages.append(
                    {"role": "assistant", "content": str(message.content)}
                )
            else:
                formatted_messages.append(
                    {"role": message.type, "content": str(message.content)}
                )
        # Use a hash of the messages as the item_id for tracking
        messages_hash = self.hash_messages(human_messages_for_hash)
        all_kwargs = {
            **self.model_kwargs,
            **({"stop": stop} if stop else {}),
            **kwargs,
        }
        # Create the batch task using BatchManager's method
        task = self.batch_manager.create_batch_task(
            custom_id=messages_hash,
            messages=formatted_messages,
            system_prompt=system_prompt,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.model_kwargs.get("max_tokens", 5000),
            **all_kwargs,
        )
        # Write to batch file
        with open(self.batch_manager.file_name, "a") as f:
            f.write(json.dumps(task) + "\n")
        generation = ChatGeneration(
            message=BaseMessage(type="ai", content=task["custom_id"])
        )
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async version that simply calls the sync version."""
        # For simplicity, just call the synchronous version
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            **self.model_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "batch-openai-chat"
