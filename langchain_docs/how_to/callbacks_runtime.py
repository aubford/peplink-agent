#!/usr/bin/env python
# coding: utf-8

# # How to pass callbacks in at runtime
# 
# :::info Prerequisites
# 
# This guide assumes familiarity with the following concepts:
# 
# - [Callbacks](/docs/concepts/callbacks)
# - [Custom callback handlers](/docs/how_to/custom_callbacks)
# 
# :::
# 
# In many cases, it is advantageous to pass in handlers instead when running the object. When we pass through [`CallbackHandlers`](https://python.langchain.com/api_reference/core/callbacks/langchain_core.callbacks.base.BaseCallbackHandler.html#langchain-core-callbacks-base-basecallbackhandler) using the `callbacks` keyword arg when executing a run, those callbacks will be issued by all nested objects involved in the execution. For example, when a handler is passed through to an Agent, it will be used for all callbacks related to the agent and all the objects involved in the agent's execution, in this case, the Tools and LLM.
# 
# This prevents us from having to manually attach the handlers to each individual nested object. Here's an example:

# In[1]:


# | output: false
# | echo: false

get_ipython().run_line_magic('pip', 'install -qU langchain langchain_anthropic')

import getpass
import os

os.environ["ANTHROPIC_API_KEY"] = getpass.getpass()


# In[4]:


from typing import Any, Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate


class LoggingHandler(BaseCallbackHandler):
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs
    ) -> None:
        print("Chat model started")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print(f"Chat model ended, response: {response}")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        print(f"Chain {serialized.get('name')} started")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        print(f"Chain ended, outputs: {outputs}")


callbacks = [LoggingHandler()]
llm = ChatAnthropic(model="claude-3-sonnet-20240229")
prompt = ChatPromptTemplate.from_template("What is 1 + {number}?")

chain = prompt | llm

chain.invoke({"number": "2"}, config={"callbacks": callbacks})


# If there are already existing callbacks associated with a module, these will run in addition to any passed in at runtime.
# 
# ## Next steps
# 
# You've now learned how to pass callbacks at runtime.
# 
# Next, check out the other how-to guides in this section, such as how to [pass callbacks into a module constructor](/docs/how_to/custom_callbacks).
