#!/usr/bin/env python
# coding: utf-8

# # DashScope
#
# Let's load the DashScope Embedding class.

# In[ ]:


from langchain_community.embeddings import DashScopeEmbeddings


# In[ ]:


embeddings = DashScopeEmbeddings(
    model="text-embedding-v1", dashscope_api_key="your-dashscope-api-key"
)


# In[ ]:


text = "This is a test document."


# In[ ]:


query_result = embeddings.embed_query(text)
print(query_result)


# In[ ]:


doc_results = embeddings.embed_documents(["foo"])
print(doc_results)
