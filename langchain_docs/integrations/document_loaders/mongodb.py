#!/usr/bin/env python
# coding: utf-8

# # MongoDB

# [MongoDB](https://www.mongodb.com/) is a NoSQL , document-oriented database that supports JSON-like documents with a dynamic schema.

# ## Overview

# The MongoDB Document Loader returns a list of Langchain Documents from a MongoDB database.
# 
# The Loader requires the following parameters:
# 
# *   MongoDB connection string
# *   MongoDB database name
# *   MongoDB collection name
# *   (Optional) Content Filter dictionary
# *   (Optional) List of field names to include in the output
# 
# The output takes the following format:
# 
# - pageContent= Mongo Document
# - metadata=\{'database': '[database_name]', 'collection': '[collection_name]'\}

# ## Load the Document Loader

# In[2]:


# add this import for running in jupyter notebook
import nest_asyncio

nest_asyncio.apply()


# In[6]:


from langchain_community.document_loaders.mongodb import MongodbLoader


# In[11]:


loader = MongodbLoader(
    connection_string="mongodb://localhost:27017/",
    db_name="sample_restaurants",
    collection_name="restaurants",
    filter_criteria={"borough": "Bronx", "cuisine": "Bakery"},
    field_names=["name", "address"],
)


# In[12]:


docs = loader.load()

len(docs)


# In[13]:


docs[0]

