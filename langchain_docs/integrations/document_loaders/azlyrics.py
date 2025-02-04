#!/usr/bin/env python
# coding: utf-8

# # AZLyrics
# 
# >[AZLyrics](https://www.azlyrics.com/) is a large, legal, every day growing collection of lyrics.
# 
# This covers how to load AZLyrics webpages into a document format that we can use downstream.

# In[1]:


from langchain_community.document_loaders import AZLyricsLoader


# In[2]:


loader = AZLyricsLoader("https://www.azlyrics.com/lyrics/mileycyrus/flowers.html")


# In[3]:


data = loader.load()


# In[4]:


data


# In[ ]:




