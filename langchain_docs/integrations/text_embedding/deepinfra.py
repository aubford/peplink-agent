#!/usr/bin/env python
# coding: utf-8

# # DeepInfra
# 
# [DeepInfra](https://deepinfra.com/?utm_source=langchain) is a serverless inference as a service that provides access to a [variety of LLMs](https://deepinfra.com/models?utm_source=langchain) and [embeddings models](https://deepinfra.com/models?type=embeddings&utm_source=langchain). This notebook goes over how to use LangChain with DeepInfra for text embeddings.

# In[1]:


# sign up for an account: https://deepinfra.com/login?utm_source=langchain

from getpass import getpass

DEEPINFRA_API_TOKEN = getpass()


# In[2]:


import os

os.environ["DEEPINFRA_API_TOKEN"] = DEEPINFRA_API_TOKEN


# In[3]:


from langchain_community.embeddings import DeepInfraEmbeddings


# In[4]:


embeddings = DeepInfraEmbeddings(
    model_id="sentence-transformers/clip-ViT-B-32",
    query_instruction="",
    embed_instruction="",
)


# In[5]:


docs = ["Dog is not a cat", "Beta is the second letter of Greek alphabet"]
document_result = embeddings.embed_documents(docs)


# In[6]:


query = "What is the first letter of Greek alphabet"
query_result = embeddings.embed_query(query)


# In[7]:


import numpy as np

query_numpy = np.array(query_result)
for doc_res, doc in zip(document_result, docs):
    document_numpy = np.array(doc_res)
    similarity = np.dot(query_numpy, document_numpy) / (
        np.linalg.norm(query_numpy) * np.linalg.norm(document_numpy)
    )
    print(f'Cosine similarity between "{doc}" and query: {similarity}')

