#!/usr/bin/env python
# coding: utf-8

# # Embedchain
# 
# >[Embedchain](https://github.com/embedchain/embedchain) is a RAG framework to create data pipelines. It loads, indexes, retrieves and syncs all the data.
# >
# >It is available as an [open source package](https://github.com/embedchain/embedchain) and as a [hosted platform solution](https://app.embedchain.ai/).
# 
# This notebook shows how to use a retriever that uses `Embedchain`.

# # Installation
# 
# First you will need to install the [`embedchain` package](https://pypi.org/project/embedchain/). 
# 
# You can install the package by running 

# In[1]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  embedchain')


# # Create New Retriever
# 
# `EmbedchainRetriever` has a static `.create()` factory method that takes the following arguments:
# 
# * `yaml_path: string` optional -- Path to the YAML configuration file. If not provided, a default configuration is used. You can browse the [docs](https://docs.embedchain.ai/) to explore various customization options.

# In[2]:


# Setup API Key

import os
from getpass import getpass

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass()


# In[3]:


from langchain_community.retrievers import EmbedchainRetriever

# create a retriever with default options
retriever = EmbedchainRetriever.create()

# or if you want to customize, pass the yaml config path
# retriever = EmbedchainRetiever.create(yaml_path="config.yaml")


# # Add Data
# 
# In embedchain, you can as many supported data types as possible. You can browse our [docs](https://docs.embedchain.ai/) to see the data types supported.
# 
# Embedchain automatically deduces the types of the data. So you can add a string, URL or local file path.

# In[4]:


retriever.add_texts(
    [
        "https://en.wikipedia.org/wiki/Elon_Musk",
        "https://www.forbes.com/profile/elon-musk",
        "https://www.youtube.com/watch?v=RcYjXbSJBN8",
    ]
)


# # Use Retriever
# 
# You can now use the retrieve to find relevant documents given a query

# In[5]:


result = retriever.invoke("How many companies does Elon Musk run and name those?")


# In[6]:


result


# In[ ]:




