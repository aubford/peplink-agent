#!/usr/bin/env python
# coding: utf-8

# # Gutenberg
# 
# >[Project Gutenberg](https://www.gutenberg.org/about/) is an online library of free eBooks.
# 
# This notebook covers how to load links to `Gutenberg` e-books into a document format that we can use downstream.

# In[1]:


from langchain_community.document_loaders import GutenbergLoader


# In[2]:


loader = GutenbergLoader("https://www.gutenberg.org/cache/epub/69972/pg69972.txt")


# In[3]:


data = loader.load()


# In[7]:


data[0].page_content[:300]


# In[9]:


data[0].metadata

