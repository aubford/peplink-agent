#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: PullMdLoader
---
# # PullMdLoader
# 
# Loader for converting URLs into Markdown using the pull.md service.
# 
# This package implements a [document loader](/docs/concepts/document_loaders/) for web content. Unlike traditional web scrapers, PullMdLoader can handle web pages built with dynamic JavaScript frameworks like React, Angular, or Vue.js, converting them into Markdown without local rendering.
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | JS Support |
# | :--- | :--- | :---: | :---: | :---: |
# | PullMdLoader | langchain-pull-md | ✅ | ✅ | ❌ |
# 

# ## Setup
# 
# ### Installation
# 
# ```bash
# pip install langchain-pull-md
# ```

# ### Initialization

# In[10]:


from langchain_pull_md.markdown_loader import PullMdLoader

# Instantiate the loader with a URL
loader = PullMdLoader(url="https://example.com")


# ### Load

# In[11]:


documents = loader.load()


# In[12]:


documents[0].metadata


# ## Lazy Load
# 
# No lazy loading is implemented. `PullMdLoader` performs a real-time conversion of the provided URL into Markdown format whenever the `load` method is called.

# ## API reference:
# 
# - [GitHub](https://github.com/chigwell/langchain-pull-md)
# - [PyPi](https://pypi.org/project/langchain-pull-md/)
