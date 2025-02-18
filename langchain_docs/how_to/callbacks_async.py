#!/usr/bin/env python
# coding: utf-8

# # How to use callbacks in async environments
#
# :::info Prerequisites
#
# This guide assumes familiarity with the following concepts:
#
# - [Callbacks](/docs/concepts/callbacks)
# - [Custom callback handlers](/docs/how_to/custom_callbacks)
# :::
#
# If you are planning to use the async APIs, it is recommended to use and extend [`AsyncCallbackHandler`](https://python.langchain.com/api_reference/core/callbacks/langchain_core.callbacks.base.AsyncCallbackHandler.html) to avoid blocking the event.
#
#
# :::warning
# If you use a sync `CallbackHandler` while using an async method to run your LLM / Chain / Tool / Agent, it will still work. However, under the hood, it will be called with [`run_in_executor`](https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor) which can cause issues if your `CallbackHandler` is not thread-safe.
# :::
#
# :::danger
#
# If you're on `python<=3.10`, you need to remember to propagate `config` or `callbacks` when invoking other `runnable` from within a `RunnableLambda`, `RunnableGenerator` or `@tool`. If you do not do this,
# the callbacks will not be propagated to the child runnables being invoked.
# :::

# In[ ]:


# | output: false
# | echo: false

get_ipython().run_line_magic("pip", "install -qU langchain langchain_anthropic")

import getpass
import os

os.environ["ANTHROPIC_API_KEY"] = getpass.getpass()


# In[2]:


import asyncio
from typing import Any, Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult


class MyCustomSyncHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"Sync handler being called in a `thread_pool_executor`: token: {token}")


class MyCustomAsyncHandler(AsyncCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        print("zzzz....")
        await asyncio.sleep(0.3)
        class_name = serialized["name"]
        print("Hi! I just woke up. Your llm is starting")

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when chain ends running."""
        print("zzzz....")
        await asyncio.sleep(0.3)
        print("Hi! I just woke up. Your llm is ending")


# To enable streaming, we pass in `streaming=True` to the ChatModel constructor
# Additionally, we pass in a list with our custom handler
chat = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    max_tokens=25,
    streaming=True,
    callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()],
)

await chat.agenerate([[HumanMessage(content="Tell me a joke")]])


# ## Next steps
#
# You've now learned how to create your own custom callback handlers.
#
# Next, check out the other how-to guides in this section, such as [how to attach callbacks to a runnable](/docs/how_to/callbacks_attach).
