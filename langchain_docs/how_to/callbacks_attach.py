#!/usr/bin/env python
# coding: utf-8

# # How to attach callbacks to a runnable
# 
# :::info Prerequisites
# 
# This guide assumes familiarity with the following concepts:
# 
# - [Callbacks](/docs/concepts/callbacks)
# - [Custom callback handlers](/docs/how_to/custom_callbacks)
# - [Chaining runnables](/docs/how_to/sequence)
# - [Attach runtime arguments to a Runnable](/docs/how_to/binding)
# 
# :::
# 
# If you are composing a chain of runnables and want to reuse callbacks across multiple executions, you can attach callbacks with the [`.with_config()`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.with_config) method. This saves you the need to pass callbacks in each time you invoke the chain.
# 
# :::important
# 
# `with_config()` binds a configuration which will be interpreted as **runtime** configuration. So these callbacks will propagate to all child components.
# :::
# 
# Here's an example:

# In[ ]:


# | output: false
# | echo: false

get_ipython().run_line_magic('pip', 'install -qU langchain langchain_anthropic')

import getpass
import os

os.environ["ANTHROPIC_API_KEY"] = getpass.getpass()


# In[1]:


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

chain_with_callbacks = chain.with_config(callbacks=callbacks)

chain_with_callbacks.invoke({"number": "2"})


# The bound callbacks will run for all nested module runs.
# 
# ## Next steps
# 
# You've now learned how to attach callbacks to a chain.
# 
# Next, check out the other how-to guides in this section, such as how to [pass callbacks in at runtime](/docs/how_to/callbacks_runtime).
