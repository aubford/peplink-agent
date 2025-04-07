#!/usr/bin/env python
# coding: utf-8

# # NLP Cloud
# 
# >[NLP Cloud](https://docs.nlpcloud.com/#introduction) is an artificial intelligence platform that allows you to use the most advanced AI engines, and even train your own engines with your own data. 
# 
# The [embeddings](https://docs.nlpcloud.com/#embeddings) endpoint offers the following model:
# 
# * `paraphrase-multilingual-mpnet-base-v2`: Paraphrase Multilingual MPNet Base V2 is a very fast model based on Sentence Transformers that is perfectly suited for embeddings extraction in more than 50 languages (see the full list here).

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  nlpcloud')


# In[1]:


from langchain_community.embeddings import NLPCloudEmbeddings


# In[2]:


import os

os.environ["NLPCLOUD_API_KEY"] = "xxx"
nlpcloud_embd = NLPCloudEmbeddings()


# In[3]:


text = "This is a test document."


# In[4]:


query_result = nlpcloud_embd.embed_query(text)


# In[5]:


doc_result = nlpcloud_embd.embed_documents([text])

