#!/usr/bin/env python
# coding: utf-8

# # Xorbits Pandas DataFrame
#
# This notebook goes over how to load data from a [xorbits.pandas](https://doc.xorbits.io/en/latest/reference/pandas/frame.html) DataFrame.

# In[1]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  xorbits")


# In[2]:


import xorbits.pandas as pd


# In[3]:


df = pd.read_csv("example_data/mlb_teams_2012.csv")


# In[9]:


df.head()


# In[5]:


from langchain_community.document_loaders import XorbitsLoader


# In[6]:


loader = XorbitsLoader(df, page_content_column="Team")


# In[7]:


loader.load()


# In[8]:


# Use lazy load for larger table, which won't read the full table into memory
for i in loader.lazy_load():
    print(i)
