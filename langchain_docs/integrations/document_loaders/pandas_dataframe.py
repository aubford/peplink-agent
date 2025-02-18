#!/usr/bin/env python
# coding: utf-8

# # Pandas DataFrame
#
# This notebook goes over how to load data from a [pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/index) DataFrame.

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  pandas")


# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("example_data/mlb_teams_2012.csv")


# In[3]:


df.head()


# In[4]:


from langchain_community.document_loaders import DataFrameLoader


# In[5]:


loader = DataFrameLoader(df, page_content_column="Team")


# In[6]:


loader.load()


# In[7]:


# Use lazy load for larger table, which won't read the full table into memory
for i in loader.lazy_load():
    print(i)
