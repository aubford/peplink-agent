#!/usr/bin/env python
# coding: utf-8
---
sidebar_class_name: hidden
---
# # Airbyte JSON (Deprecated)

# Note: `AirbyteJSONLoader` is deprecated. Please use [`AirbyteLoader`](/docs/integrations/document_loaders/airbyte) instead.
# 
# >[Airbyte](https://github.com/airbytehq/airbyte) is a data integration platform for ELT pipelines from APIs, databases & files to warehouses & lakes. It has the largest catalog of ELT connectors to data warehouses and databases.

# This covers how to load any source from Airbyte into a local JSON file that can be read in as a document
# 
# Prereqs:
# Have docker desktop installed
# 
# Steps:
# 
# 1) Clone Airbyte from GitHub - `git clone https://github.com/airbytehq/airbyte.git`
# 
# 2) Switch into Airbyte directory - `cd airbyte`
# 
# 3) Start Airbyte - `docker compose up`
# 
# 4) In your browser, just visit http://localhost:8000. You will be asked for a username and password. By default, that's username `airbyte` and password `password`.
# 
# 5) Setup any source you wish.
# 
# 6) Set destination as Local JSON, with specified destination path - lets say `/json_data`. Set up manual sync.
# 
# 7) Run the connection.
# 
# 7) To see what files are create, you can navigate to: `file:///tmp/airbyte_local`
# 
# 8) Find your data and copy path. That path should be saved in the file variable below. It should start with `/tmp/airbyte_local`
# 

# In[1]:


from langchain_community.document_loaders import AirbyteJSONLoader


# In[2]:


get_ipython().system('ls /tmp/airbyte_local/json_data/')


# In[3]:


loader = AirbyteJSONLoader("/tmp/airbyte_local/json_data/_airbyte_raw_pokemon.jsonl")


# In[4]:


data = loader.load()


# In[8]:


print(data[0].page_content[:500])


# In[ ]:




