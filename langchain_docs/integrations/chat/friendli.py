#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: ChatFriendli
---
# # ChatFriendli
# 
# > [Friendli](https://friendli.ai/) enhances AI application performance and optimizes cost savings with scalable, efficient deployment options, tailored for high-demand AI workloads.
# 
# This tutorial guides you through integrating `ChatFriendli` for chat applications using LangChain. `ChatFriendli` offers a flexible approach to generating conversational AI responses, supporting both synchronous and asynchronous calls.

# ## Setup
# 
# Ensure the `langchain_community` and `friendli-client` are installed.
# 
# ```sh
# pip install -U langchain-community friendli-client.
# ```
# 
# Sign in to [Friendli Suite](https://suite.friendli.ai/) to create a Personal Access Token, and set it as the `FRIENDLI_TOKEN` environment.

# In[2]:


import getpass
import os

if "FRIENDLI_TOKEN" not in os.environ:
    os.environ["FRIENDLI_TOKEN"] = getpass.getpass("Friendi Personal Access Token: ")


# You can initialize a Friendli chat model with selecting the model you want to use. The default model is `mixtral-8x7b-instruct-v0-1`. You can check the available models at [docs.friendli.ai](https://docs.periflow.ai/guides/serverless_endpoints/pricing#text-generation-models).

# In[3]:


from langchain_community.chat_models.friendli import ChatFriendli

chat = ChatFriendli(model="meta-llama-3.1-8b-instruct", max_tokens=100, temperature=0)


# ## Usage
# 
# `FrienliChat` supports all methods of [`ChatModel`](/docs/how_to#chat-models) including async APIs.

# You can also use functionality of  `invoke`, `batch`, `generate`, and `stream`.

# In[4]:


from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage

system_message = SystemMessage(content="Answer questions as short as you can.")
human_message = HumanMessage(content="Tell me a joke.")
messages = [system_message, human_message]

chat.invoke(messages)


# In[5]:


chat.batch([messages, messages])


# In[6]:


chat.generate([messages, messages])


# In[8]:


for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)


# You can also use all functionality of async APIs: `ainvoke`, `abatch`, `agenerate`, and `astream`.

# In[9]:


await chat.ainvoke(messages)


# In[10]:


await chat.abatch([messages, messages])


# In[11]:


await chat.agenerate([messages, messages])


# In[12]:


async for chunk in chat.astream(messages):
    print(chunk.content, end="", flush=True)

