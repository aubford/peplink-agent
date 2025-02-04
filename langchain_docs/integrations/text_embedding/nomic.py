#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Nomic
---
# # NomicEmbeddings
# 
# This will help you get started with Nomic embedding models using LangChain. For detailed documentation on `NomicEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/nomic/embeddings/langchain_nomic.embeddings.NomicEmbeddings.html).
# 
# ## Overview
# ### Integration details
# 
# import { ItemTable } from "@theme/FeatureTables";
# 
# <ItemTable category="text_embedding" item="Nomic" />
# 
# ## Setup
# 
# To access Nomic embedding models you'll need to create a/an Nomic account, get an API key, and install the `langchain-nomic` integration package.
# 
# ### Credentials
# 
# Head to [https://atlas.nomic.ai/](https://atlas.nomic.ai/) to sign up to Nomic and generate an API key. Once you've done this set the `NOMIC_API_KEY` environment variable:

# In[2]:


import getpass
import os

if not os.getenv("NOMIC_API_KEY"):
    os.environ["NOMIC_API_KEY"] = getpass.getpass("Enter your Nomic API key: ")


# If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[3]:


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")


# ### Installation
# 
# The LangChain Nomic integration lives in the `langchain-nomic` package:

# In[2]:


get_ipython().run_line_magic('pip', 'install -qU langchain-nomic')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:

# In[10]:


from langchain_nomic import NomicEmbeddings

embeddings = NomicEmbeddings(
    model="nomic-embed-text-v1.5",
    # dimensionality=256,
    # Nomic's `nomic-embed-text-v1.5` model was [trained with Matryoshka learning](https://blog.nomic.ai/posts/nomic-embed-matryoshka)
    # to enable variable-length embeddings with a single model.
    # This means that you can specify the dimensionality of the embeddings at inference time.
    # The model supports dimensionality from 64 to 768.
    # inference_mode="remote",
    # One of `remote`, `local` (Embed4All), or `dynamic` (automatic). Defaults to `remote`.
    # api_key=... , # if using remote inference,
    # device="cpu",
    # The device to use for local embeddings. Choices include
    # `cpu`, `gpu`, `nvidia`, `amd`, or a specific device name. See
    # the docstring for `GPT4All.__init__` for more info. Typically
    # defaults to CPU. Do not use on macOS.
)


# ## Indexing and Retrieval
# 
# Embedding models are often used in retrieval-augmented generation (RAG) flows, both as part of indexing data as well as later retrieving it. For more detailed instructions, please see our [RAG tutorials](/docs/tutorials/).
# 
# Below, see how to index and retrieve data using the `embeddings` object we initialized above. In this example, we will index and retrieve a sample document in the `InMemoryVectorStore`.

# In[5]:


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

# In[6]:


single_vector = embeddings.embed_query(text)
print(str(single_vector)[:100])  # Show the first 100 characters of the vector


# ### Embed multiple texts
# 
# You can embed multiple texts with `embed_documents`:

# In[7]:


text2 = (
    "LangGraph is a library for building stateful, multi-actor applications with LLMs"
)
two_vectors = embeddings.embed_documents([text, text2])
for vector in two_vectors:
    print(str(vector)[:100])  # Show the first 100 characters of the vector


# ## API Reference
# 
# For detailed documentation on `NomicEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/nomic/embeddings/langchain_nomic.embeddings.NomicEmbeddings.html).
# 
