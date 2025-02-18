#!/usr/bin/env python
# coding: utf-8

# # AsyncHtml
#
# `AsyncHtmlLoader` loads raw HTML from a list of URLs concurrently.

# In[4]:


from langchain_community.document_loaders import AsyncHtmlLoader


# In[5]:


urls = ["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/"]
loader = AsyncHtmlLoader(urls)
# If you need to use the proxy to make web requests, for example using http_proxy/https_proxy environmental variables,
# please set trust_env=True explicitly here as follows:
# loader = AsyncHtmlLoader(urls, trust_env=True)
# Otherwise, loader.load() may stuck becuase aiohttp session does not recognize the proxy by default
docs = loader.load()


# In[6]:


docs[0].page_content[1000:2000]


# In[7]:


docs[1].page_content[1000:2000]
