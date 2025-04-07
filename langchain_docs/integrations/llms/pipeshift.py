#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Pipeshift
---
# # Pipeshift
# 
# This will help you get started with Pipeshift completion models (LLMs) using LangChain. For detailed documentation on `Pipeshift` features and configuration options, please refer to the [API reference](https://dashboard.pipeshift.com/docs).
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/llms/pipeshift) | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [Pipeshift](https://dashboard.pipeshift.com/docs) | [langchain-pipeshift](https://pypi.org/project/langchain-pipeshift/) | ❌ | - | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-pipeshift?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-pipeshift?style=flat-square&label=%20) |
# 
# ## Setup
# 
# To access Pipeshift models you'll need to create a Pipeshift account, get an API key, and install the `langchain-pipeshift` integration package.
# 
# ### Credentials
# 
# Head to [Pipeshift](https://dashboard.pipeshift.com) to sign up to Pipeshift and generate an API key. Once you've done this set the PIPESHIFT_API_KEY environment variable:

# In[1]:


import getpass
import os

if not os.getenv("PIPESHIFT_API_KEY"):
    os.environ["PIPESHIFT_API_KEY"] = getpass.getpass("Enter your Pipeshift API key: ")


# If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[2]:


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")


# ### Installation
# 
# The LangChain Pipeshift integration lives in the `langchain-pipeshift` package:

# In[3]:


get_ipython().run_line_magic('pip', 'install -qU langchain-pipeshift')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:

# In[4]:


from langchain_pipeshift import Pipeshift

llm = Pipeshift(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    temperature=0,
    max_tokens=512,
)


# ## Invocation

# In[5]:


input_text = "Pipeshift is an AI company that "

completion = llm.invoke(input_text)
completion


# ## Chaining
# 
# We can also [chain](/docs/how_to/sequence/) our llm with a prompt template

# ## API reference
# 
# For detailed documentation of all `Pipeshift` features and configurations head to the API reference: https://dashboard.pipeshift.com/docs 
