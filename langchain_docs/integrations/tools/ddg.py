#!/usr/bin/env python
# coding: utf-8

# # DuckDuckGo Search
#
# This guide shows over how to use the DuckDuckGo search component.
#
# ## Usage

# In[1]:


get_ipython().run_line_magic("pip", "install -qU duckduckgo-search langchain-community")


# In[2]:


from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

search.invoke("Obama's first name?")


# To get more additional information (e.g. link, source) use `DuckDuckGoSearchResults()`

# In[3]:


from langchain_community.tools import DuckDuckGoSearchResults

search = DuckDuckGoSearchResults()

search.invoke("Obama")


# By default the results are returned as a comma-separated string of key-value pairs from the original search results. You can also choose to return the search results as a list by setting `output_format="list"` or as a JSON string by setting `output_format="json"`.

# In[4]:


search = DuckDuckGoSearchResults(output_format="list")

search.invoke("Obama")


# You can also just search for news articles. Use the keyword `backend="news"`

# In[5]:


search = DuckDuckGoSearchResults(backend="news")

search.invoke("Obama")


# You can also directly pass a custom `DuckDuckGoSearchAPIWrapper` to `DuckDuckGoSearchResults` to provide more control over the search results.

# In[6]:


from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)

search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")

search.invoke("Obama")


# ## Related
#
# - [How to use a chat model to call tools](https://python.langchain.com/docs/how_to/tool_calling/)
