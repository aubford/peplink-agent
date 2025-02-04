#!/usr/bin/env python
# coding: utf-8

# # Polars DataFrame
# 
# This notebook goes over how to load data from a [polars](https://pola-rs.github.io/polars-book/user-guide/) DataFrame.

# In[1]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  polars')


# In[2]:


import polars as pl


# In[3]:


df = pl.read_csv("example_data/mlb_teams_2012.csv")


# In[4]:


df.head()


# In[5]:


from langchain_community.document_loaders import PolarsDataFrameLoader


# In[6]:


loader = PolarsDataFrameLoader(df, page_content_column="Team")


# In[7]:


loader.load()


# In[8]:


# Use lazy load for larger table, which won't read the full table into memory
for i in loader.lazy_load():
    print(i)

