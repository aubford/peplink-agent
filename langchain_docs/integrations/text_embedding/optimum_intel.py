#!/usr/bin/env python
# coding: utf-8

# # Embedding Documents using Optimized and Quantized Embedders
# 
# Embedding all documents using Quantized Embedders.
# 
# The embedders are based on optimized models, created by using [optimum-intel](https://github.com/huggingface/optimum-intel.git) and [IPEX](https://github.com/intel/intel-extension-for-pytorch).
# 
# Example text is based on [SBERT](https://www.sbert.net/docs/pretrained_cross-encoders.html).

# In[2]:


from langchain_community.embeddings import QuantizedBiEncoderEmbeddings

model_name = "Intel/bge-small-en-v1.5-rag-int8-static"
encode_kwargs = {"normalize_embeddings": True}  # set True to compute cosine similarity

model = QuantizedBiEncoderEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs,
    query_instruction="Represent this sentence for searching relevant passages: ",
)


# Lets ask a question, and compare to 2 documents. The first contains the answer to the question, and the second one does not. 
# 
# We can check better suits our query.

# In[5]:


question = "How many people live in Berlin?"


# In[6]:


documents = [
    "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "Berlin is well known for its museums.",
]


# In[7]:


doc_vecs = model.embed_documents(documents)


# In[8]:


query_vec = model.embed_query(question)


# In[10]:


import torch


# In[11]:


doc_vecs_torch = torch.tensor(doc_vecs)


# In[12]:


query_vec_torch = torch.tensor(query_vec)


# In[15]:


query_vec_torch @ doc_vecs_torch.T


# We can see that indeed the first one ranks higher.
