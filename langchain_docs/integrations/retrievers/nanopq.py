#!/usr/bin/env python
# coding: utf-8

# # NanoPQ (Product Quantization)
# 
# >[Product Quantization algorithm (k-NN)](https://towardsdatascience.com/similarity-search-product-quantization-b2a1a6397701) in brief is a quantization algorithm that helps in compression of database vectors which helps in semantic search when large datasets are involved. In a nutshell, the embedding is split into M subspaces which further goes through clustering. Upon clustering the vectors the centroid vector gets mapped to the vectors present in the each of the clusters of the subspace. 
# 
# This notebook goes over how to use a retriever that under the hood uses a Product Quantization which has been implemented by the [nanopq](https://github.com/matsui528/nanopq) package.

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-community langchain-openai nanopq')


# In[1]:


from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.retrievers import NanoPQRetriever


# ## Create New Retriever with Texts

# In[2]:


retriever = NanoPQRetriever.from_texts(
    ["Great world", "great words", "world", "planets of the world"],
    SpacyEmbeddings(model_name="en_core_web_sm"),
    clusters=2,
    subspace=2,
)


# ## Use Retriever
# 
# We can now use the retriever!

# In[3]:


retriever.invoke("earth")


# In[ ]:




