#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: ZhipuAI
keywords: [zhipuaiembeddings]
---
# # ZhipuAIEmbeddings
# 
# This will help you get started with ZhipuAI embedding models using LangChain. For detailed documentation on `ZhipuAIEmbeddings` features and configuration options, please refer to the [API reference](https://bigmodel.cn/dev/api#vector).
# 
# ## Overview
# ### Integration details
# 
# | Provider | Package |
# |:--------:|:-------:|
# | [ZhipuAI](/docs/integrations/providers/zhipuai/) | [langchain-community](https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.zhipuai.ZhipuAIEmbeddings.html) |
# 
# ## Setup
# 
# To access ZhipuAI embedding models you'll need to create a/an ZhipuAI account, get an API key, and install the `zhipuai` integration package.
# 
# ### Credentials
# 
# Head to [https://bigmodel.cn/](https://bigmodel.cn/usercenter/apikeys) to sign up to ZhipuAI and generate an API key. Once you've done this set the ZHIPUAI_API_KEY environment variable:

# In[1]:


import getpass
import os

if not os.getenv("ZHIPUAI_API_KEY"):
    os.environ["ZHIPUAI_API_KEY"] = getpass.getpass("Enter your ZhipuAI API key: ")


# If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[2]:


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")


# ### Installation
# 
# The LangChain ZhipuAI integration lives in the `zhipuai` package:

# In[3]:


get_ipython().run_line_magic('pip', 'install -qU zhipuai')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:

# In[4]:


from langchain_community.embeddings import ZhipuAIEmbeddings

embeddings = ZhipuAIEmbeddings(
    model="embedding-3",
    # With the `embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
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

# In[9]:


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
# For detailed documentation on `ZhipuAIEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.zhipuai.ZhipuAIEmbeddings.html).
# 
