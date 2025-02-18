#!/usr/bin/env python
# coding: utf-8

# # Merge Documents Loader
#
# Merge the documents returned from a set of specified data loaders.

# In[1]:


from langchain_community.document_loaders import WebBaseLoader

loader_web = WebBaseLoader(
    "https://github.com/basecamp/handbook/blob/master/37signals-is-you.md"
)


# In[2]:


from langchain_community.document_loaders import PyPDFLoader

loader_pdf = PyPDFLoader("../MachineLearning-Lecture01.pdf")


# In[3]:


from langchain_community.document_loaders.merge import MergedDataLoader

loader_all = MergedDataLoader(loaders=[loader_web, loader_pdf])


# In[4]:


docs_all = loader_all.load()


# In[8]:


len(docs_all)
