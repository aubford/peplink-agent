#!/usr/bin/env python
# coding: utf-8
---
sidebar_position: 1.5
---
# # How to stream chat model responses
# 
# 
# All [chat models](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html) implement the [Runnable interface](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable), which comes with a **default** implementations of standard runnable methods (i.e. `ainvoke`, `batch`, `abatch`, `stream`, `astream`, `astream_events`).
# 
# The **default** streaming implementation provides an`Iterator` (or `AsyncIterator` for asynchronous streaming) that yields a single value: the final output from the underlying chat model provider.
# 
# :::tip
# 
# The **default** implementation does **not** provide support for token-by-token streaming, but it ensures that the the model can be swapped in for any other model as it supports the same standard interface.
# 
# :::
# 
# The ability to stream the output token-by-token depends on whether the provider has implemented proper streaming support.
# 
# See which [integrations support token-by-token streaming here](/docs/integrations/chat/).

# ## Sync streaming
# 
# Below we use a `|` to help visualize the delimiter between tokens.

# In[1]:


from langchain_anthropic.chat_models import ChatAnthropic

chat = ChatAnthropic(model="claude-3-haiku-20240307")
for chunk in chat.stream("Write me a 1 verse song about goldfish on the moon"):
    print(chunk.content, end="|", flush=True)


# ## Async Streaming

# In[2]:


from langchain_anthropic.chat_models import ChatAnthropic

chat = ChatAnthropic(model="claude-3-haiku-20240307")
async for chunk in chat.astream("Write me a 1 verse song about goldfish on the moon"):
    print(chunk.content, end="|", flush=True)


# ## Astream events
# 
# Chat models also support the standard [astream events](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.astream_events) method.
# 
# This method is useful if you're streaming output from a larger LLM application that contains multiple steps (e.g., an LLM chain composed of a prompt, llm and parser).

# In[3]:


from langchain_anthropic.chat_models import ChatAnthropic

chat = ChatAnthropic(model="claude-3-haiku-20240307")
idx = 0

async for event in chat.astream_events(
    "Write me a 1 verse song about goldfish on the moon"
):
    idx += 1
    if idx >= 5:  # Truncate the output
        print("...Truncated")
        break
    print(event)

