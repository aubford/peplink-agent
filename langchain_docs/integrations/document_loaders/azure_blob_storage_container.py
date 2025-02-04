#!/usr/bin/env python
# coding: utf-8

# # Azure Blob Storage Container
# 
# >[Azure Blob Storage](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blobs-introduction) is Microsoft's object storage solution for the cloud. Blob Storage is optimized for storing massive amounts of unstructured data. Unstructured data is data that doesn't adhere to a particular data model or definition, such as text or binary data.
# 
# `Azure Blob Storage` is designed for:
# - Serving images or documents directly to a browser.
# - Storing files for distributed access.
# - Streaming video and audio.
# - Writing to log files.
# - Storing data for backup and restore, disaster recovery, and archiving.
# - Storing data for analysis by an on-premises or Azure-hosted service.
# 
# This notebook covers how to load document objects from a container on `Azure Blob Storage`.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  azure-storage-blob')


# In[2]:


from langchain_community.document_loaders import AzureBlobStorageContainerLoader


# In[3]:


loader = AzureBlobStorageContainerLoader(conn_str="<conn_str>", container="<container>")


# In[4]:


loader.load()


# ## Specifying a prefix
# You can also specify a prefix for more finegrained control over what files to load.

# In[5]:


loader = AzureBlobStorageContainerLoader(
    conn_str="<conn_str>", container="<container>", prefix="<prefix>"
)


# In[6]:


loader.load()


# In[ ]:




