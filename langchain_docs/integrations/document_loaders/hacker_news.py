#!/usr/bin/env python
# coding: utf-8

# # Hacker News
#
# >[Hacker News](https://en.wikipedia.org/wiki/Hacker_News) (sometimes abbreviated as `HN`) is a social news website focusing on computer science and entrepreneurship. It is run by the investment fund and startup incubator `Y Combinator`. In general, content that can be submitted is defined as "anything that gratifies one's intellectual curiosity."
#
# This notebook covers how to pull page data and comments from [Hacker News](https://news.ycombinator.com/)

# In[1]:


from langchain_community.document_loaders import HNLoader


# In[2]:


loader = HNLoader("https://news.ycombinator.com/item?id=34817881")


# In[3]:


data = loader.load()


# In[4]:


data[0].page_content[:300]


# In[5]:


data[0].metadata
