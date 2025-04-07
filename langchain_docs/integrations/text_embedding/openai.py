#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: OpenAI
keywords: [openaiembeddings]
---
# # OpenAIEmbeddings
# 
# This will help you get started with OpenAI embedding models using LangChain. For detailed documentation on `OpenAIEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/openai/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html).
# 
# 
# ## Overview
# ### Integration details
# 
# import { ItemTable } from "@theme/FeatureTables";
# 
# <ItemTable category="text_embedding" item="OpenAI" />
# 
# ## Setup
# 
# To access OpenAI embedding models you'll need to create a/an OpenAI account, get an API key, and install the `langchain-openai` integration package.
# 
# ### Credentials
# 
# Head to [platform.openai.com](https://platform.openai.com) to sign up to OpenAI and generate an API key. Once you’ve done this set the OPENAI_API_KEY environment variable:

# In[6]:


import getpass
import os

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


# To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:

# In[7]:


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")


# ### Installation
# 
# The LangChain OpenAI integration lives in the `langchain-openai` package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-openai')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:

# In[10]:


from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
)


# ## Indexing and Retrieval
# 
# Embedding models are often used in retrieval-augmented generation (RAG) flows, both as part of indexing data as well as later retrieving it. For more detailed instructions, please see our [RAG tutorials](/docs/tutorials/).
# 
# Below, see how to index and retrieve data using the `embeddings` object we initialized above. In this example, we will index and retrieve a sample document in the `InMemoryVectorStore`.

# In[11]:


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

# In[12]:


single_vector = embeddings.embed_query(text)
print(str(single_vector)[:100])  # Show the first 100 characters of the vector


# ### Embed multiple texts
# 
# You can embed multiple texts with `embed_documents`:

# In[13]:


text2 = (
    "LangGraph is a library for building stateful, multi-actor applications with LLMs"
)
two_vectors = embeddings.embed_documents([text, text2])
for vector in two_vectors:
    print(str(vector)[:100])  # Show the first 100 characters of the vector


# ## API Reference
# 
# For detailed documentation on `OpenAIEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/openai/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html).
# 
