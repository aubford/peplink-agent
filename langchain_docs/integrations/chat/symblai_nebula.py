#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Nebula (Symbl.ai)
---
# # Nebula (Symbl.ai)
# 
# ## Overview
# This notebook covers how to get started with [Nebula](https://docs.symbl.ai/docs/nebula-llm) - Symbl.ai's chat model.
# 
# ### Integration details
# Head to the [API reference](https://docs.symbl.ai/reference/nebula-chat) for detailed documentation.
# 
# ### Model features: TODO

# ## Setup
# 
# ### Credentials
# To get started, request a [Nebula API key](https://platform.symbl.ai/#/login) and set the `NEBULA_API_KEY` environment variable:

# In[2]:


import getpass
import os

os.environ["NEBULA_API_KEY"] = getpass.getpass()


# ### Installation
# The integration is set up in the `langchain-community` package.

# ## Instantiation

# In[ ]:


from langchain_community.chat_models.symblai_nebula import ChatNebula
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# In[4]:


chat = ChatNebula(max_tokens=1024, temperature=0.5)


# ## Invocation

# In[5]:


messages = [
    SystemMessage(
        content="You are a helpful assistant that answers general knowledge questions."
    ),
    HumanMessage(content="What is the capital of France?"),
]
chat.invoke(messages)


# ### Async

# In[6]:


await chat.ainvoke(messages)


# ### Streaming

# In[9]:


for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)


# ### Batch

# In[12]:


chat.batch([messages])


# ## Chaining

# In[18]:


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | chat


# In[19]:


chain.invoke({"topic": "cows"})


# ## API reference
# 
# Check out the [API reference](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.symblai_nebula.ChatNebula.html) for more detail.
