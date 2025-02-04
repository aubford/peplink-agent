#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: LiteLLM
---
# # ChatLiteLLM
# 
# [LiteLLM](https://github.com/BerriAI/litellm) is a library that simplifies calling Anthropic, Azure, Huggingface, Replicate, etc. 
# 
# This notebook covers how to get started with using Langchain + the LiteLLM I/O library. 

# In[1]:


from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage


# In[2]:


chat = ChatLiteLLM(model="gpt-3.5-turbo")


# In[3]:


messages = [
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    )
]
chat(messages)


# ## `ChatLiteLLM` also supports async and streaming functionality:

# In[4]:


from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler


# In[5]:


await chat.agenerate([messages])


# In[6]:


chat = ChatLiteLLM(
    streaming=True,
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)
chat(messages)


# In[ ]:




