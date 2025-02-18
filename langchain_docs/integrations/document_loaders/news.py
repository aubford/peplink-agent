#!/usr/bin/env python
# coding: utf-8

# # News URL
#
# This covers how to load HTML news articles from a list of URLs into a document format that we can use downstream.

# In[1]:


from langchain_community.document_loaders import NewsURLLoader


# In[2]:


urls = [
    "https://www.bbc.com/news/world-us-canada-66388172",
    "https://www.bbc.com/news/entertainment-arts-66384971",
]


# Pass in urls to load them into Documents

# In[3]:


loader = NewsURLLoader(urls=urls)
data = loader.load()
print("First article: ", data[0])
print("\nSecond article: ", data[1])


# Use nlp=True to run nlp analysis and generate keywords + summary

# In[4]:


loader = NewsURLLoader(urls=urls, nlp=True)
data = loader.load()
print("First article: ", data[0])
print("\nSecond article: ", data[1])


# In[5]:


data[0].metadata["keywords"]


# In[6]:


data[0].metadata["summary"]
