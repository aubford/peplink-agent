#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: LiteLLM Router
---
# 

# # ChatLiteLLMRouter
# 
# [LiteLLM](https://github.com/BerriAI/litellm) is a library that simplifies calling Anthropic, Azure, Huggingface, Replicate, etc. 
# 
# This notebook covers how to get started with using Langchain + the LiteLLM Router I/O library. 

# In[1]:


from langchain_community.chat_models import ChatLiteLLMRouter
from langchain_core.messages import HumanMessage
from litellm import Router


# In[2]:


model_list = [
    {
        "model_name": "gpt-4",
        "litellm_params": {
            "model": "azure/gpt-4-1106-preview",
            "api_key": "<your-api-key>",
            "api_version": "2023-05-15",
            "api_base": "https://<your-endpoint>.openai.azure.com/",
        },
    },
    {
        "model_name": "gpt-35-turbo",
        "litellm_params": {
            "model": "azure/gpt-35-turbo",
            "api_key": "<your-api-key>",
            "api_version": "2023-05-15",
            "api_base": "https://<your-endpoint>.openai.azure.com/",
        },
    },
]
litellm_router = Router(model_list=model_list)
chat = ChatLiteLLMRouter(router=litellm_router, model_name="gpt-35-turbo")


# In[3]:


messages = [
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    )
]
chat(messages)


# ## `ChatLiteLLMRouter` also supports async and streaming functionality:

# In[4]:


from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler


# In[5]:


await chat.agenerate([messages])


# In[6]:


chat = ChatLiteLLMRouter(
    router=litellm_router,
    model_name="gpt-35-turbo",
    streaming=True,
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)
chat(messages)


# In[ ]:




