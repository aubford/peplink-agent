#!/usr/bin/env python
# coding: utf-8

# # IMSDb
# 
# >[IMSDb](https://imsdb.com/) is the `Internet Movie Script Database`.
# 
# This covers how to load `IMSDb` webpages into a document format that we can use downstream.

# In[1]:


from langchain_community.document_loaders import IMSDbLoader


# In[2]:


loader = IMSDbLoader("https://imsdb.com/scripts/BlacKkKlansman.html")


# In[3]:


data = loader.load()


# In[8]:


data[0].page_content[:500]


# In[6]:


data[0].metadata

