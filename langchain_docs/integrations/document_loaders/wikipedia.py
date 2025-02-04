#!/usr/bin/env python
# coding: utf-8

# # Wikipedia
# 
# >[Wikipedia](https://wikipedia.org/) is a multilingual free online encyclopedia written and maintained by a community of volunteers, known as Wikipedians, through open collaboration and using a wiki-based editing system called MediaWiki. `Wikipedia` is the largest and most-read reference work in history.
# 
# This notebook shows how to load wiki pages from `wikipedia.org` into the Document format that we use downstream.

# ## Installation
# 
# First, you need to install the `langchain_community` and `wikipedia` packages.

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain_community wikipedia')


# ## Parameters
# 
# `WikipediaLoader` has the following arguments:
# - `query`: the free text which used to find documents in Wikipedia
# - `lang` (optional): default="en". Use it to search in a specific language part of Wikipedia
# - `load_max_docs` (optional): default=100. Use it to limit number of downloaded documents. It takes time to download all 100 documents, so use a small number for experiments. There is a hard limit of 300 for now.
# - `load_all_available_meta` (optional): default=False. By default only the most important fields downloaded: `title` and `summary`. If `True` then all available fields will be downloaded.
# - `doc_content_chars_max` (optional): default=4000. The maximum number of characters for the document content.

# ## Example

# In[1]:


from langchain_community.document_loaders import WikipediaLoader


# In[2]:


docs = WikipediaLoader(query="HUNTER X HUNTER", load_max_docs=2).load()
len(docs)


# In[3]:


docs[0].metadata  # metadata of the first document


# In[4]:


docs[0].page_content[:400]  # a part of the page content

