#!/usr/bin/env python
# coding: utf-8

# # Fake Embeddings
# 
# LangChain also provides a fake embedding class. You can use this to test your pipelines.

# In[1]:


from langchain_community.embeddings import FakeEmbeddings


# In[3]:


embeddings = FakeEmbeddings(size=1352)


# In[5]:


query_result = embeddings.embed_query("foo")


# In[6]:


doc_results = embeddings.embed_documents(["foo"])

