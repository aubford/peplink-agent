#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: YoutubeLoaderDL
---
# # YoutubeLoaderDL
# 
# Loader for Youtube leveraging the `yt-dlp` library.
# 
# This package implements a [document loader](/docs/concepts/document_loaders/) for Youtube. In contrast to the [YoutubeLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.youtube.YoutubeLoader.html) of `langchain-community`, which relies on `pytube`, `YoutubeLoaderDL` is able to fetch YouTube metadata. `langchain-yt-dlp` leverages the robust `yt-dlp` library, providing a more reliable and feature-rich YouTube document loader.
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | JS Support |
# | :--- | :--- | :---: | :---: | :---: |
# | YoutubeLoader | langchain-yt-dlp | ✅ | ✅ | ❌ |
# 
# ## Setup
# 
# ### Installation
# 
# ```bash
# pip install langchain-yt-dlp
# ```

# ### Initialization

# In[10]:


from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL

# Basic transcript loading
loader = YoutubeLoaderDL.from_youtube_url(
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ", add_video_info=True
)


# ### Load

# In[11]:


documents = loader.load()


# In[12]:


documents[0].metadata


# ## Lazy Load

# - No lazy loading is implemented

# ## API reference:
# 
# - [Github](https://github.com/aqib0770/langchain-yt-dlp)
# - [PyPi](https://pypi.org/project/langchain-yt-dlp/)
