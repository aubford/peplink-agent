#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: ModelScope
---
# # ModelScopeEmbeddings
# 
# ModelScope ([Home](https://www.modelscope.cn/) | [GitHub](https://github.com/modelscope/modelscope)) is built upon the notion of “Model-as-a-Service” (MaaS). It seeks to bring together most advanced machine learning models from the AI community, and streamlines the process of leveraging AI models in real-world applications. The core ModelScope library open-sourced in this repository provides the interfaces and implementations that allow developers to perform model inference, training and evaluation. 
# 
# This will help you get started with ModelScope embedding models using LangChain.
# 
# ## Overview
# ### Integration details
# 
# | Provider | Package |
# |:--------:|:-------:|
# | [ModelScope](/docs/integrations/providers/modelscope/) | [langchain-modelscope-integration](https://pypi.org/project/langchain-modelscope-integration/) |
# 
# ## Setup
# 
# To access ModelScope embedding models you'll need to create a/an ModelScope account, get an API key, and install the `langchain-modelscope-integration` integration package.
# 
# ### Credentials
# 
# Head to [ModelScope](https://modelscope.cn/) to sign up to ModelScope.

# In[ ]:


import getpass
import os

if not os.getenv("MODELSCOPE_SDK_TOKEN"):
    os.environ["MODELSCOPE_SDK_TOKEN"] = getpass.getpass(
        "Enter your ModelScope SDK token: "
    )


# ### Installation
# 
# The LangChain ModelScope integration lives in the `langchain-modelscope-integration` package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-modelscope-integration')


# ## Instantiation
# 
# Now we can instantiate our model object:

# In[3]:


from langchain_modelscope import ModelScopeEmbeddings

embeddings = ModelScopeEmbeddings(
    model_id="damo/nlp_corom_sentence-embedding_english-base",
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
# For detailed documentation on `ModelScopeEmbeddings` features and configuration options, please refer to the [API reference](https://www.modelscope.cn/docs/sdk/pipelines).
# 
