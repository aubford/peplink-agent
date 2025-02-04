#!/usr/bin/env python
# coding: utf-8

# # Dropbox
# 
# [Dropbox](https://en.wikipedia.org/wiki/Dropbox) is a file hosting service that brings everything-traditional files, cloud content, and web shortcuts together in one place.
# 
# This notebook covers how to load documents from *Dropbox*. In addition to common files such as text and PDF files, it also supports *Dropbox Paper* files.
# 
# ## Prerequisites
# 
# 1. Create a Dropbox app.
# 2. Give the app these scope permissions: `files.metadata.read` and `files.content.read`.
# 3. Generate access token: https://www.dropbox.com/developers/apps/create.
# 4. `pip install dropbox` (requires `pip install "unstructured[pdf]"` for PDF filetype).
# 
# ## Instructions
# 
# `DropboxLoader`` requires you to create a Dropbox App and generate an access token. This can be done from https://www.dropbox.com/developers/apps/create. You also need to have the Dropbox Python SDK installed (pip install dropbox).
# 
# DropboxLoader can load data from a list of Dropbox file paths or a single Dropbox folder path. Both paths should be relative to the root directory of the Dropbox account linked to the access token.

# In[1]:


pip install dropbox


# In[2]:


from langchain_community.document_loaders import DropboxLoader


# In[3]:


# Generate access token: https://www.dropbox.com/developers/apps/create.
dropbox_access_token = "<DROPBOX_ACCESS_TOKEN>"
# Dropbox root folder
dropbox_folder_path = ""


# In[4]:


loader = DropboxLoader(
    dropbox_access_token=dropbox_access_token,
    dropbox_folder_path=dropbox_folder_path,
    recursive=False,
)


# In[5]:


documents = loader.load()


# In[ ]:


for document in documents:
    print(document)

