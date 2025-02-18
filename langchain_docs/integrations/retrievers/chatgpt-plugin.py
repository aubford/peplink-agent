#!/usr/bin/env python
# coding: utf-8

# # ChatGPT plugin
#
# >[OpenAI plugins](https://platform.openai.com/docs/plugins/introduction) connect `ChatGPT` to third-party applications. These plugins enable `ChatGPT` to interact with APIs defined by developers, enhancing `ChatGPT's` capabilities and allowing it to perform a wide range of actions.
#
# >Plugins allow `ChatGPT` to do things like:
# >- Retrieve real-time information; e.g., sports scores, stock prices, the latest news, etc.
# >- Retrieve knowledge-base information; e.g., company docs, personal notes, etc.
# >- Perform actions on behalf of the user; e.g., booking a flight, ordering food, etc.
#
# This notebook shows how to use the ChatGPT Retriever Plugin within LangChain.

# In[2]:


# STEP 1: Load

# Load documents using LangChain's DocumentLoaders
# This is from https://langchain.readthedocs.io/en/latest/modules/document_loaders/examples/csv.html

from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document

loader = CSVLoader(
    file_path="../../document_loaders/examples/example_data/mlb_teams_2012.csv"
)
data = loader.load()


# STEP 2: Convert

# Convert Document to format expected by https://github.com/openai/chatgpt-retrieval-plugin
import json
from typing import List


def write_json(path: str, documents: List[Document]) -> None:
    results = [{"text": doc.page_content} for doc in documents]
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


write_json("foo.json", data)

# STEP 3: Use

# Ingest this as you would any other json file in https://github.com/openai/chatgpt-retrieval-plugin/tree/main/scripts/process_json


# ## Using the ChatGPT Retriever Plugin
#
# Okay, so we've created the ChatGPT Retriever Plugin, but how do we actually use it?
#
# The below code walks through how to do that.

# We want to use `ChatGPTPluginRetriever` so we have to get the OpenAI API Key.

# In[6]:


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")


# In[7]:


from langchain_community.retrievers import (
    ChatGPTPluginRetriever,
)


# In[10]:


retriever = ChatGPTPluginRetriever(url="http://0.0.0.0:8000", bearer_token="foo")


# In[3]:


retriever.invoke("alice's phone number")


# In[ ]:
