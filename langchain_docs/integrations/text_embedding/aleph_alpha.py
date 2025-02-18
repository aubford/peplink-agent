#!/usr/bin/env python
# coding: utf-8

# # Aleph Alpha
#
# There are two possible ways to use Aleph Alpha's semantic embeddings. If you have texts with a dissimilar structure (e.g. a Document and a Query) you would want to use asymmetric embeddings. Conversely, for texts with comparable structures, symmetric embeddings are the suggested approach.

# ## Asymmetric

# In[1]:


from langchain_community.embeddings import AlephAlphaAsymmetricSemanticEmbedding


# In[2]:


document = "This is a content of the document"
query = "What is the content of the document?"


# In[3]:


embeddings = AlephAlphaAsymmetricSemanticEmbedding(normalize=True, compress_to_size=128)


# In[4]:


doc_result = embeddings.embed_documents([document])


# In[5]:


query_result = embeddings.embed_query(query)


# ## Symmetric

# In[6]:


from langchain_community.embeddings import AlephAlphaSymmetricSemanticEmbedding


# In[7]:


text = "This is a test text"


# In[8]:


embeddings = AlephAlphaSymmetricSemanticEmbedding(normalize=True, compress_to_size=128)


# In[9]:


doc_result = embeddings.embed_documents([text])


# In[10]:


query_result = embeddings.embed_query(text)


# In[ ]:
