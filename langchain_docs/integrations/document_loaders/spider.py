#!/usr/bin/env python
# coding: utf-8

# # Spider
# [Spider](https://spider.cloud/) is the [fastest](https://github.com/spider-rs/spider/blob/main/benches/BENCHMARKS.md) and most affordable crawler and scraper that returns LLM-ready data.
# 
# ## Setup

# In[ ]:


pip install spider-client


# ## Usage
# To use spider you need to have an API key from [spider.cloud](https://spider.cloud/).

# In[2]:


from langchain_community.document_loaders import SpiderLoader

loader = SpiderLoader(
    api_key="YOUR_API_KEY",
    url="https://spider.cloud",
    mode="scrape",  # if no API key is provided it looks for SPIDER_API_KEY in env
)

data = loader.load()
print(data)


# ## Modes
# - `scrape`: Default mode that scrapes a single URL
# - `crawl`: Crawl all subpages of the domain url provided

# ## Crawler options
# The `params` parameter is a dictionary that can be passed to the loader. See the [Spider documentation](https://spider.cloud/docs/api) to see all available parameters
