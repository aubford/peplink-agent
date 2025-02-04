#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: GPTRouter
---
# # GPTRouter
# 
# [GPTRouter](https://github.com/Writesonic/GPTRouter) is an open source LLM API Gateway that offers a universal API for 30+ LLMs, vision, and image models, with smart fallbacks based on uptime and latency, automatic retries, and streaming.
# 
#  
# This notebook covers how to get started with using Langchain + the GPTRouter I/O library. 
# 
# * Set `GPT_ROUTER_API_KEY` environment variable
# * or use the `gpt_router_api_key` keyword argument

# In[14]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  GPTRouter')


# In[15]:


from langchain_community.chat_models import GPTRouter
from langchain_community.chat_models.gpt_router import GPTRouterModel
from langchain_core.messages import HumanMessage


# In[16]:


anthropic_claude = GPTRouterModel(name="claude-instant-1.2", provider_name="anthropic")


# In[17]:


chat = GPTRouter(models_priority_list=[anthropic_claude])


# In[18]:


messages = [
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    )
]
chat(messages)


# ## `GPTRouter` also supports async and streaming functionality:

# In[19]:


from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler


# In[20]:


await chat.agenerate([messages])


# In[21]:


chat = GPTRouter(
    models_priority_list=[anthropic_claude],
    streaming=True,
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)
chat(messages)

