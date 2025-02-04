#!/usr/bin/env python
# coding: utf-8

# # Brave Search
# 
# This notebook goes over how to use the Brave Search tool.
# Go to the [Brave Website](https://brave.com/search/api/) to sign up for a free account and get an API key.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet langchain-community')


# In[ ]:


from langchain_community.tools import BraveSearch


# In[2]:


api_key = "API KEY"


# In[3]:


tool = BraveSearch.from_api_key(api_key=api_key, search_kwargs={"count": 3})


# In[4]:


tool.run("obama middle name")


# In[ ]:




