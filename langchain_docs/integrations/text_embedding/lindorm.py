#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Lindorm
---
# # LindormAIEmbeddings
# 
# This will help you get started with Lindorm embedding models using LangChain. 
# 
# ## Overview
# ### Integration details
# 
# | Provider |              Package              |
# |:--------:|:---------------------------------:|
# | [Lindorm](/docs/integrations/providers/lindorm/) | [langchain-lindorm-integration](https://pypi.org/project/langchain-lindorm-integration/) |
# 
# ## Setup
# 
# 
# To access Lindorm embedding models you'll need to create a Lindorm account, get AK&SK, and install the `langchain-lindorm-integration` integration package.
# 
# ### Credentials
# 
# 
# You can get you credentials in the [console](https://lindorm.console.aliyun.com/cn-hangzhou/clusterhou/cluster?spm=a2c4g.11186623.0.0.466534e93Xj6tt)
# 

# In[1]:


import os


class Config:
    AI_LLM_ENDPOINT = os.environ.get("AI_ENDPOINT", "<AI_ENDPOINT>")
    AI_USERNAME = os.environ.get("AI_USERNAME", "root")
    AI_PWD = os.environ.get("AI_PASSWORD", "<PASSWORD>")

    AI_DEFAULT_EMBEDDING_MODEL = "bge_m3_model"  # set to your deployed model


# ### Installation
# 
# The LangChain Lindorm integration lives in the `langchain-lindorm-integration` package:

# In[2]:


get_ipython().run_line_magic('pip', 'install -qU langchain-lindorm-integration')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:
# 

# In[3]:


from langchain_lindorm_integration import LindormAIEmbeddings

embeddings = LindormAIEmbeddings(
    endpoint=Config.AI_LLM_ENDPOINT,
    username=Config.AI_USERNAME,
    password=Config.AI_PWD,
    model_name=Config.AI_DEFAULT_EMBEDDING_MODEL,
)


# ## Indexing and Retrieval
# 
# Embedding models are often used in retrieval-augmented generation (RAG) flows, both as part of indexing data as well as later retrieving it. For more detailed instructions, please see our [RAG tutorials](/docs/tutorials/).
# 
# Below, see how to index and retrieve data using the `embeddings` object we initialized above. In this example, we will index and retrieve a sample document in the `InMemoryVectorStore`.

# In[4]:


# Create a vector store with a sample text
from langchain_core.vectorstores import InMemoryVectorStore

text = "LangChain is the framework for building context-aware reasoning applications"

vectorstore = InMemoryVectorStore.from_texts(
    [text],
    embedding=embeddings,
)

# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever()

# Retrieve the most similar text
retrieved_documents = retriever.invoke("What is LangChain?")

# show the retrieved document's content
retrieved_documents[0].page_content


# ## Direct Usage
# 
# Under the hood, the vectorstore and retriever implementations are calling `embeddings.embed_documents(...)` and `embeddings.embed_query(...)` to create embeddings for the text(s) used in `from_texts` and retrieval `invoke` operations, respectively.
# 
# You can directly call these methods to get embeddings for your own use cases.
# 
# ### Embed single texts
# 
# You can embed single texts or documents with `embed_query`:

# In[5]:


single_vector = embeddings.embed_query(text)
print(str(single_vector)[:100])  # Show the first 100 characters of the vector


# ### Embed multiple texts
# 
# You can embed multiple texts with `embed_documents`:

# In[6]:


text2 = (
    "LangGraph is a library for building stateful, multi-actor applications with LLMs"
)
two_vectors = embeddings.embed_documents([text, text2])
for vector in two_vectors:
    print(str(vector)[:100])  # Show the first 100 characters of the vector


# ## API Reference
# 
# For detailed documentation on `LindormEmbeddings` features and configuration options, please refer to the [API reference](https://pypi.org/project/langchain-lindorm-integration/).
# 
