#!/usr/bin/env python
# coding: utf-8

# # ReadTheDocs Documentation
# 
# >[Read the Docs](https://readthedocs.org/) is an open-sourced free software documentation hosting platform. It generates documentation written with the `Sphinx` documentation generator.
# 
# This notebook covers how to load content from HTML that was generated as part of a `Read-The-Docs` build.
# 
# For an example of this in the wild, see [here](https://github.com/langchain-ai/chat-langchain).
# 
# This assumes that the HTML has already been scraped into a folder. This can be done by uncommenting and running the following command

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  beautifulsoup4')


# In[ ]:


#!wget -r -A.html -P rtdocs https://python.langchain.com/en/latest/


# In[1]:


from langchain_community.document_loaders import ReadTheDocsLoader


# In[3]:


loader = ReadTheDocsLoader("rtdocs")


# In[ ]:


docs = loader.load()

