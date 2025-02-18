#!/usr/bin/env python
# coding: utf-8

# # Google Trends
#
# This notebook goes over how to use the Google Trends Tool to fetch trends information.
#
# First, you need to sign up for an `SerpApi key` key at: https://serpapi.com/users/sign_up.
#
# Then you must install `google-search-results` with the command:
#
# `pip install google-search-results`
#
# Then you will need to set the environment variable `SERPAPI_API_KEY` to your `SerpApi key`
#
# [Alternatively you can pass the key in as a argument to the wrapper `serp_api_key="your secret key"`]
#
# ## Use the Tool

# In[1]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet  google-search-results langchain_community"
)


# In[2]:


import os

from langchain_community.tools.google_trends import GoogleTrendsQueryRun
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper

os.environ["SERPAPI_API_KEY"] = ""
tool = GoogleTrendsQueryRun(api_wrapper=GoogleTrendsAPIWrapper())


# In[5]:


tool.run("Water")
