#!/usr/bin/env python
# coding: utf-8

# # MiniMax
# 
# [MiniMax](https://api.minimax.chat/document/guides/embeddings?id=6464722084cdc277dfaa966a) offers an embeddings service.
# 
# This example goes over how to use LangChain to interact with MiniMax Inference for text embedding.

# In[1]:


import os

os.environ["MINIMAX_GROUP_ID"] = "MINIMAX_GROUP_ID"
os.environ["MINIMAX_API_KEY"] = "MINIMAX_API_KEY"


# In[2]:


from langchain_community.embeddings import MiniMaxEmbeddings


# In[3]:


embeddings = MiniMaxEmbeddings()


# In[4]:


query_text = "This is a test query."
query_result = embeddings.embed_query(query_text)


# In[5]:


document_text = "This is a test document."
document_result = embeddings.embed_documents([document_text])


# In[6]:


import numpy as np

query_numpy = np.array(query_result)
document_numpy = np.array(document_result[0])
similarity = np.dot(query_numpy, document_numpy) / (
    np.linalg.norm(query_numpy) * np.linalg.norm(document_numpy)
)
print(f"Cosine similarity between document and query: {similarity}")


# In[ ]:




