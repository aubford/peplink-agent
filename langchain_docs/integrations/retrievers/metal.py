#!/usr/bin/env python
# coding: utf-8

# # Metal
#
# >[Metal](https://github.com/getmetal/metal-python) is a managed service for ML Embeddings.
#
# This notebook shows how to use [Metal's](https://docs.getmetal.io/introduction) retriever.
#
# First, you will need to sign up for Metal and get an API key. You can do so [here](https://docs.getmetal.io/misc-create-app)

# In[1]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  metal_sdk")


# In[3]:


from metal_sdk.metal import Metal

API_KEY = ""
CLIENT_ID = ""
INDEX_ID = ""

metal = Metal(API_KEY, CLIENT_ID, INDEX_ID)


# ## Ingest Documents
#
# You only need to do this if you haven't already set up an index

# In[8]:


metal.index({"text": "foo1"})
metal.index({"text": "foo"})


# ## Query
#
# Now that our index is set up, we can set up a retriever and start querying it.

# In[9]:


from langchain_community.retrievers import MetalRetriever


# In[10]:


retriever = MetalRetriever(metal, params={"limit": 2})


# In[11]:


retriever.invoke("foo1")


# In[ ]:
