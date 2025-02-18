#!/usr/bin/env python
# coding: utf-8

# # Azure Blob Storage File
#
# >[Azure Files](https://learn.microsoft.com/en-us/azure/storage/files/storage-files-introduction) offers fully managed file shares in the cloud that are accessible via the industry standard Server Message Block (`SMB`) protocol, Network File System (`NFS`) protocol, and `Azure Files REST API`.
#
# This covers how to load document objects from a Azure Files.

# In[1]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  azure-storage-blob")


# In[1]:


from langchain_community.document_loaders import AzureBlobStorageFileLoader


# In[8]:


loader = AzureBlobStorageFileLoader(
    conn_str="<connection string>",
    container="<container name>",
    blob_name="<blob name>",
)


# In[9]:


loader.load()


# In[ ]:
