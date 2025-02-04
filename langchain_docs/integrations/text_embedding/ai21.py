#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: AI21
---
# # AI21Embeddings
# 
# :::caution This service is deprecated. :::
# 
# This will help you get started with AI21 embedding models using LangChain. For detailed documentation on `AI21Embeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/ai21/embeddings/langchain_ai21.embeddings.AI21Embeddings.html).
# 
# ## Overview
# ### Integration details
# 
# import { ItemTable } from "@theme/FeatureTables";
# 
# <ItemTable category="text_embedding" item="AI21" />
# 
# ## Setup
# 
# To access AI21 embedding models you'll need to create an AI21 account, get an API key, and install the `langchain-ai21` integration package.
# 
# ### Credentials
# 
# Head to [https://docs.ai21.com/](https://docs.ai21.com/) to sign up to AI21 and generate an API key. Once you've done this set the `AI21_API_KEY` environment variable:

# In[2]:


import getpass
import os

if not os.getenv("AI21_API_KEY"):
    os.environ["AI21_API_KEY"] = getpass.getpass("Enter your AI21 API key: ")


# If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[3]:


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")


# ### Installation
# 
# The LangChain AI21 integration lives in the `langchain-ai21` package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-ai21')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:

# In[7]:


from langchain_ai21 import AI21Embeddings

embeddings = AI21Embeddings(
    # Can optionally increase or decrease the batch_size
    # to improve latency.
    # Use larger batch sizes with smaller documents, and
    # smaller batch sizes with larger documents.
    # batch_size=256,
)


# ## Indexing and Retrieval
# 
# Embedding models are often used in retrieval-augmented generation (RAG) flows, both as part of indexing data as well as later retrieving it. For more detailed instructions, please see our [RAG tutorials](/docs/tutorials/).
# 
# Below, see how to index and retrieve data using the `embeddings` object we initialized above. In this example, we will index and retrieve a sample document in the `InMemoryVectorStore`.

# In[8]:


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

# In[9]:


single_vector = embeddings.embed_query(text)
print(str(single_vector)[:100])  # Show the first 100 characters of the vector


# ### Embed multiple texts
# 
# You can embed multiple texts with `embed_documents`:

# In[10]:


text2 = (
    "LangGraph is a library for building stateful, multi-actor applications with LLMs"
)
two_vectors = embeddings.embed_documents([text, text2])
for vector in two_vectors:
    print(str(vector)[:100])  # Show the first 100 characters of the vector


# ## API Reference
# 
# For detailed documentation on `AI21Embeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/ai21/embeddings/langchain_ai21.embeddings.AI21Embeddings.html).
# 
