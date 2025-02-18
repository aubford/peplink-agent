#!/usr/bin/env python
# coding: utf-8

# # Facebook Chat
#
# >[Messenger](https://en.wikipedia.org/wiki/Messenger_(software)) is an American proprietary instant messaging app and platform developed by `Meta Platforms`. Originally developed as `Facebook Chat` in 2008, the company revamped its messaging service in 2010.
#
# This notebook covers how to load data from the [Facebook Chats](https://www.facebook.com/business/help/1646890868956360) into a format that can be ingested into LangChain.

# In[ ]:


# pip install pandas


# In[1]:


from langchain_community.document_loaders import FacebookChatLoader


# In[2]:


loader = FacebookChatLoader("example_data/facebook_chat.json")


# In[7]:


loader.load()
