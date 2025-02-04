#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Friendli
---
# # Friendli
# 
# > [Friendli](https://friendli.ai/) enhances AI application performance and optimizes cost savings with scalable, efficient deployment options, tailored for high-demand AI workloads.
# 
# This tutorial guides you through integrating `Friendli` with LangChain.

# ## Setup
# 
# Ensure the `langchain_community` and `friendli-client` are installed.
# 
# ```sh
# pip install -U langchain-community friendli-client
# ```
# 
# Sign in to [Friendli Suite](https://suite.friendli.ai/) to create a Personal Access Token, and set it as the `FRIENDLI_TOKEN` environment.

# In[1]:


import getpass
import os

if "FRIENDLI_TOKEN" not in os.environ:
    os.environ["FRIENDLI_TOKEN"] = getpass.getpass("Friendi Personal Access Token: ")


# You can initialize a Friendli chat model with selecting the model you want to use.  
# The default model is `meta-llama-3.1-8b-instruct`. You can check the available models at [friendli.ai/docs](https://friendli.ai/docs/guides/serverless_endpoints/pricing#text-generation-models).

# In[2]:


from langchain_community.llms.friendli import Friendli

llm = Friendli(model="meta-llama-3.1-8b-instruct", max_tokens=100, temperature=0)


# ## Usage
# 
# `Frienli` supports all methods of [`LLM`](/docs/how_to#llms) including async APIs.

# You can use functionality of `invoke`, `batch`, `generate`, and `stream`.

# In[3]:


llm.invoke("Tell me a joke.")


# In[4]:


llm.batch(["Tell me a joke.", "Tell me a joke."])


# In[5]:


llm.generate(["Tell me a joke.", "Tell me a joke."])


# In[ ]:


for chunk in llm.stream("Tell me a joke."):
    print(chunk, end="", flush=True)


# You can also use all functionality of async APIs: `ainvoke`, `abatch`, `agenerate`, and `astream`.

# In[6]:


await llm.ainvoke("Tell me a joke.")


# In[7]:


await llm.abatch(["Tell me a joke.", "Tell me a joke."])


# In[8]:


await llm.agenerate(["Tell me a joke.", "Tell me a joke."])


# In[ ]:


async for chunk in llm.astream("Tell me a joke."):
    print(chunk, end="", flush=True)

