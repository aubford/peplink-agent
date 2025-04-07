#!/usr/bin/env python
# coding: utf-8

# # LocalAI
# 
# :::info
# 
# `langchain-localai` is a 3rd party integration package for LocalAI. It provides a simple way to use LocalAI services in Langchain.
# 
# The source code is available on [Github](https://github.com/mkhludnev/langchain-localai)
# 
# :::
# 
# Let's load the LocalAI Embedding class. In order to use the LocalAI Embedding class, you need to have the LocalAI service hosted somewhere and configure the embedding models. See the documentation at https://localai.io/basics/getting_started/index.html and https://localai.io/features/embeddings/index.html.

# In[ ]:


get_ipython().run_line_magic('pip', 'install -U langchain-localai')


# In[2]:


from langchain_localai import LocalAIEmbeddings


# In[3]:


embeddings = LocalAIEmbeddings(
    openai_api_base="http://localhost:8080", model="embedding-model-name"
)


# In[4]:


text = "This is a test document."


# In[ ]:


query_result = embeddings.embed_query(text)


# In[5]:


doc_result = embeddings.embed_documents([text])


# Let's load the LocalAI Embedding class with first generation models (e.g. text-search-ada-doc-001/text-search-ada-query-001). Note: These are not recommended models - see [here](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings)

# In[ ]:


from langchain_community.embeddings import LocalAIEmbeddings


# In[ ]:


embeddings = LocalAIEmbeddings(
    openai_api_base="http://localhost:8080", model="embedding-model-name"
)


# In[ ]:


text = "This is a test document."


# In[ ]:


query_result = embeddings.embed_query(text)


# In[ ]:


doc_result = embeddings.embed_documents([text])


# In[ ]:


import os

# if you are behind an explicit proxy, you can use the OPENAI_PROXY environment variable to pass through
os.environ["OPENAI_PROXY"] = "http://proxy.yourcompany.com:8080"

