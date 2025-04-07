#!/usr/bin/env python
# coding: utf-8

# # Instruct Embeddings on Hugging Face
# 
# >[Hugging Face sentence-transformers](https://huggingface.co/sentence-transformers) is a Python framework for state-of-the-art sentence, text and image embeddings.
# >One of the instruct embedding models is used in the `HuggingFaceInstructEmbeddings` class.
# 

# In[8]:


from langchain_community.embeddings import HuggingFaceInstructEmbeddings


# In[9]:


embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Represent the query for retrieval: "
)


# In[10]:


text = "This is a test document."


# In[11]:


query_result = embeddings.embed_query(text)


# In[ ]:




