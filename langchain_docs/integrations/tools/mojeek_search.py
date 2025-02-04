#!/usr/bin/env python
# coding: utf-8

# # Mojeek Search
# 
# The following notebook will explain how to get results using Mojeek Search. Please visit [Mojeek Website](https://www.mojeek.com/services/search/web-search-api/) to obtain an API key.

# In[ ]:


from langchain_community.tools import MojeekSearch


# In[ ]:


api_key = "KEY"  # obtained from Mojeek Website


# In[ ]:


search = MojeekSearch.config(api_key=api_key, search_kwargs={"t": 10})


# In `search_kwargs` you can add any search parameter that you can find on [Mojeek Documentation](https://www.mojeek.com/support/api/search/request_parameters.html)

# In[ ]:


search.run("mojeek")

