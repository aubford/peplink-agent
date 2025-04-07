#!/usr/bin/env python
# coding: utf-8

# # TensorFlow Hub
# 
# >[TensorFlow Hub](https://www.tensorflow.org/hub) is a repository of trained machine learning models ready for fine-tuning and deployable anywhere. Reuse trained models like `BERT` and `Faster R-CNN` with just a few lines of code.
# >
# >
# Let's load the TensorflowHub Embedding class.

# In[1]:


from langchain_community.embeddings import TensorflowHubEmbeddings


# In[5]:


embeddings = TensorflowHubEmbeddings()


# In[6]:


text = "This is a test document."


# In[7]:


query_result = embeddings.embed_query(text)


# In[6]:


doc_results = embeddings.embed_documents(["foo"])


# In[ ]:


doc_results


# In[ ]:




