#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Together AI
---
# # TogetherEmbeddings
# 
# This will help you get started with Together embedding models using LangChain. For detailed documentation on `TogetherEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/together/embeddings/langchain_together.embeddings.TogetherEmbeddings.html).
# 
# ## Overview
# ### Integration details
# 
# import { ItemTable } from "@theme/FeatureTables";
# 
# <ItemTable category="text_embedding" item="Together" />
# 
# ## Setup
# 
# To access Together embedding models you'll need to create a/an Together account, get an API key, and install the `langchain-together` integration package.
# 
# ### Credentials
# 
# Head to [https://api.together.xyz/](https://api.together.xyz/) to sign up to Together and generate an API key. Once you've done this set the TOGETHER_API_KEY environment variable:

# In[1]:


import getpass
import os

if not os.getenv("TOGETHER_API_KEY"):
    os.environ["TOGETHER_API_KEY"] = getpass.getpass("Enter your Together API key: ")


# To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:

# In[2]:


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")


# ### Installation
# 
# The LangChain Together integration lives in the `langchain-together` package:

# In[3]:


get_ipython().run_line_magic('pip', 'install -qU langchain-together')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:

# In[5]:


from langchain_together import TogetherEmbeddings

embeddings = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-8k-retrieval",
)


# ## Indexing and Retrieval
# 
# Embedding models are often used in retrieval-augmented generation (RAG) flows, both as part of indexing data as well as later retrieving it. For more detailed instructions, please see our [RAG tutorials](/docs/tutorials/).
# 
# Below, see how to index and retrieve data using the `embeddings` object we initialized above. In this example, we will index and retrieve a sample document in the `InMemoryVectorStore`.

# In[6]:


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

# In[7]:


single_vector = embeddings.embed_query(text)
print(str(single_vector)[:100])  # Show the first 100 characters of the vector


# ### Embed multiple texts
# 
# You can embed multiple texts with `embed_documents`:

# In[8]:


text2 = (
    "LangGraph is a library for building stateful, multi-actor applications with LLMs"
)
two_vectors = embeddings.embed_documents([text, text2])
for vector in two_vectors:
    print(str(vector)[:100])  # Show the first 100 characters of the vector


# ## API Reference
# 
# For detailed documentation on `TogetherEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/together/embeddings/langchain_together.embeddings.TogetherEmbeddings.html).
# 
