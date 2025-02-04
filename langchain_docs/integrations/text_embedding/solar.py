#!/usr/bin/env python
# coding: utf-8

# # Solar
# 
# [Solar](https://console.upstage.ai/services/embedding) offers an embeddings service.
# 
# This example goes over how to use LangChain to interact with Solar Inference for text embedding.

# In[9]:


import os

os.environ["SOLAR_API_KEY"] = ""


# In[2]:


from langchain_community.embeddings import SolarEmbeddings


# In[3]:


embeddings = SolarEmbeddings()


# In[4]:


query_text = "This is a test query."
query_result = embeddings.embed_query(query_text)


# In[5]:


query_result


# In[6]:


document_text = "This is a test document."
document_result = embeddings.embed_documents([document_text])


# In[7]:


document_result


# In[8]:


import numpy as np

query_numpy = np.array(query_result)
document_numpy = np.array(document_result[0])
similarity = np.dot(query_numpy, document_numpy) / (
    np.linalg.norm(query_numpy) * np.linalg.norm(document_numpy)
)
print(f"Cosine similarity between document and query: {similarity}")

