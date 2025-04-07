#!/usr/bin/env python
# coding: utf-8

# ## Overview
# 
# Model2Vec is a technique to turn any sentence transformer into a really small static model
# [model2vec](https://github.com/MinishLab/model2vec) can be used to generate embeddings.

# ## Setup
# 
# ```bash
# pip install -U langchain-community
# ```
# 

# ## Instantiation

# Ensure that `model2vec` is installed
# 
# ```bash
# pip install -U model2vec
# ```

# ## Indexing and Retrieval

# In[2]:


from langchain_community.embeddings import Model2vecEmbeddings


# In[3]:


embeddings = Model2vecEmbeddings("minishlab/potion-base-8M")


# In[4]:


query_text = "This is a test query."
query_result = embeddings.embed_query(query_text)


# In[6]:


document_text = "This is a test document."
document_result = embeddings.embed_documents([document_text])


# ## Direct Usage
# 
# Here's how you would directly make use of `model2vec`
# 
# ```python
# from model2vec import StaticModel
# 
# # Load a model from the HuggingFace hub (in this case the potion-base-8M model)
# model = StaticModel.from_pretrained("minishlab/potion-base-8M")
# 
# # Make embeddings
# embeddings = model.encode(["It's dangerous to go alone!", "It's a secret to everybody."])
# 
# # Make sequences of token embeddings
# token_embeddings = model.encode_as_sequence(["It's dangerous to go alone!", "It's a secret to everybody."])
# ```

# ## API Reference
# 
# For more information check out the model2vec github [repo](https://github.com/MinishLab/model2vec)
