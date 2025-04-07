#!/usr/bin/env python
# coding: utf-8

# # Anyscale
# 
# Let's load the Anyscale Embedding class.

# In[1]:


from langchain_community.embeddings import AnyscaleEmbeddings


# In[6]:


embeddings = AnyscaleEmbeddings(
    anyscale_api_key="ANYSCALE_API_KEY", model="thenlper/gte-large"
)


# In[7]:


text = "This is a test document."


# In[10]:


query_result = embeddings.embed_query(text)
print(query_result)


# In[9]:


doc_result = embeddings.embed_documents([text])
print(doc_result)

