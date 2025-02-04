#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: AstraDB
---
# # AstraDBByteStore
# 
# This will help you get started with Astra DB [key-value stores](/docs/concepts/key_value_stores). For detailed documentation of all `AstraDBByteStore` features and configurations head to the [API reference](https://python.langchain.com/api_reference/astradb/storage/langchain_astradb.storage.AstraDBByteStore.html).
# 
# ## Overview
# 
# DataStax [Astra DB](https://docs.datastax.com/en/astra/home/astra.html) is a serverless vector-capable database built on Cassandra and made conveniently available through an easy-to-use JSON API.
# 
# ### Integration details
# 
# | Class | Package | Local | JS support | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: |
# | [AstraDBByteStore](https://python.langchain.com/api_reference/astradb/storage/langchain_astradb.storage.AstraDBByteStore.html) | [langchain_astradb](https://python.langchain.com/api_reference/astradb/index.html) | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_astradb?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_astradb?style=flat-square&label=%20) |
# 
# ## Setup
# 
# To create an `AstraDBByteStore` byte store, you'll need to [create a DataStax account](https://www.datastax.com/products/datastax-astra).
# 
# ### Credentials
# 
# After signing up, set the following credentials:

# In[1]:


from getpass import getpass

ASTRA_DB_API_ENDPOINT = getpass("ASTRA_DB_API_ENDPOINT = ")
ASTRA_DB_APPLICATION_TOKEN = getpass("ASTRA_DB_APPLICATION_TOKEN = ")


# ### Installation
# 
# The LangChain AstraDB integration lives in the `langchain_astradb` package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain_astradb')


# ## Instantiation
# 
# Now we can instantiate our byte store:

# In[5]:


from langchain_astradb import AstraDBByteStore

kv_store = AstraDBByteStore(
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    collection_name="my_store",
)


# ## Usage
# 
# You can set data under keys like this using the `mset` method:

# In[6]:


kv_store.mset(
    [
        ["key1", b"value1"],
        ["key2", b"value2"],
    ]
)

kv_store.mget(
    [
        "key1",
        "key2",
    ]
)


# And you can delete data using the `mdelete` method:

# In[7]:


kv_store.mdelete(
    [
        "key1",
        "key2",
    ]
)

kv_store.mget(
    [
        "key1",
        "key2",
    ]
)


# You can use an `AstraDBByteStore` anywhere you'd use other ByteStores, including as a [cache for embeddings](/docs/how_to/caching_embeddings).

# ## API reference
# 
# For detailed documentation of all `AstraDBByteStore` features and configurations, head to the API reference: https://python.langchain.com/api_reference/astradb/storage/langchain_astradb.storage.AstraDBByteStore.html
