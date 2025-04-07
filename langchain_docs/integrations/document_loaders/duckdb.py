#!/usr/bin/env python
# coding: utf-8

# # DuckDB
# 
# >[DuckDB](https://duckdb.org/) is an in-process SQL OLAP database management system.
# 
# Load a `DuckDB` query with one document per row.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  duckdb')


# In[2]:


from langchain_community.document_loaders import DuckDBLoader


# In[3]:


get_ipython().run_cell_magic('file', 'example.csv', 'Team,Payroll\nNationals,81.34\nReds,82.20\n')


# In[4]:


loader = DuckDBLoader("SELECT * FROM read_csv_auto('example.csv')")

data = loader.load()


# In[5]:


print(data)


# ## Specifying Which Columns are Content vs Metadata

# In[5]:


loader = DuckDBLoader(
    "SELECT * FROM read_csv_auto('example.csv')",
    page_content_columns=["Team"],
    metadata_columns=["Payroll"],
)

data = loader.load()


# In[6]:


print(data)


# ## Adding Source to Metadata

# In[7]:


loader = DuckDBLoader(
    "SELECT Team, Payroll, Team As source FROM read_csv_auto('example.csv')",
    metadata_columns=["source"],
)

data = loader.load()


# In[8]:


print(data)


# In[ ]:




