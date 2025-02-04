#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Upstage
---
# # UpstageDocumentParseLoader
# 
# This notebook covers how to get started with `UpstageDocumentParseLoader`.
# 
# ## Installation
# 
# Install `langchain-upstage` package.
# 
# ```bash
# pip install -U langchain-upstage
# ```

# ## Environment Setup
# 
# Make sure to set the following environment variables:
# 
# - `UPSTAGE_API_KEY`: Your Upstage API key. Read [Upstage developers document](https://developers.upstage.ai/docs/getting-started/quick-start) to get your API key.
# 
# > The previously used UPSTAGE_DOCUMENT_AI_API_KEY is deprecated. However, the key previously used in UPSTAGE_DOCUMENT_AI_API_KEY can now be used in UPSTAGE_API_KEY.

# ## Usage

# In[ ]:


import os

os.environ["UPSTAGE_API_KEY"] = "YOUR_API_KEY"


# In[1]:


from langchain_upstage import UpstageDocumentParseLoader

file_path = "/PATH/TO/YOUR/FILE.pdf"
layzer = UpstageDocumentParseLoader(file_path, split="page")

# For improved memory efficiency, consider using the lazy_load method to load documents page by page.
docs = layzer.load()  # or layzer.lazy_load()

for doc in docs[:3]:
    print(doc)

