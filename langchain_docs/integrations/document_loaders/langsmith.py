#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: LangSmith
---
# # LangSmithLoader
# 
# This notebook provides a quick overview for getting started with the LangSmith [document loader](https://python.langchain.com/docs/concepts/document_loaders). For detailed documentation of all LangSmithLoader features and configurations head to the [API reference](https://python.langchain.com/api_reference/core/document_loaders/langchain_core.document_loaders.langsmith.LangSmithLoader.html).
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | JS support|
# | :--- | :--- | :---: | :---: |  :---: |
# | [LangSmithLoader](https://python.langchain.com/api_reference/core/document_loaders/langchain_core.document_loaders.langsmith.LangSmithLoader.html) | [langchain-core](https://python.langchain.com/api_reference/core/index.html) | ❌ | ❌ | ❌ | 
# 
# ### Loader features
# | Source | Lazy loading | Native async
# | :---: | :---: | :---: | 
# | LangSmithLoader | ✅ | ❌ | 
# 
# ## Setup
# 
# To access the LangSmith document loader you'll need to install `langchain-core`, create a [LangSmith](https://langsmith.com) account and get an API key.
# 
# ### Credentials
# 
# Sign up at https://langsmith.com and generate an API key. Once you've done this set the LANGSMITH_API_KEY environment variable:

# In[ ]:


import getpass
import os

if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")


# If you want to get automated best-in-class tracing, you can also turn on LangSmith tracing:

# In[1]:


# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# Install `langchain-core`:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-core')


# ### Clone example dataset
# 
# For this example, we'll clone and load a public LangSmith dataset. Cloning creates a copy of this dataset on our personal LangSmith account. You can only load datasets that you have a personal copy of.

# In[10]:


from langsmith import Client as LangSmithClient

ls_client = LangSmithClient()

dataset_name = "LangSmith Few Shot Datasets Notebook"
dataset_public_url = (
    "https://smith.langchain.com/public/55658626-124a-4223-af45-07fb774a6212/d"
)

ls_client.clone_public_dataset(dataset_public_url)


# ## Initialization
# 
# Now we can instantiate our document loader and load documents:

# In[11]:


from langchain_core.document_loaders import LangSmithLoader

loader = LangSmithLoader(
    dataset_name=dataset_name,
    content_key="question",
    limit=50,
    # format_content=...,
    # ...
)


# ## Load

# In[12]:


docs = loader.load()
print(docs[0].page_content)


# In[15]:


print(docs[0].metadata["inputs"])


# In[16]:


print(docs[0].metadata["outputs"])


# In[19]:


list(docs[0].metadata.keys())


# ## Lazy Load

# In[20]:


page = []
for doc in loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:
        # do some paged operation, e.g.
        # index.upsert(page)
        # page = []
        break
len(page)


# ## API reference
# 
# For detailed documentation of all LangSmithLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/core/document_loaders/langchain_core.document_loaders.langsmith.LangSmithLoader.html
