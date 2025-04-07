#!/usr/bin/env python
# coding: utf-8

# # Google Scholar
# 
# This notebook goes through how to use Google Scholar Tool

# In[5]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  google-search-results langchain-community')


# In[6]:


import os

from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper


# In[7]:


os.environ["SERP_API_KEY"] = ""
tool = GoogleScholarQueryRun(api_wrapper=GoogleScholarAPIWrapper())
tool.run("LLM Models")


# In[ ]:




