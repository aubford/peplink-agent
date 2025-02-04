#!/usr/bin/env python
# coding: utf-8

# # PyPDFLoader
# 
# This notebook provides a quick overview for getting started with `PyPDF` [document loader](https://python.langchain.com/docs/concepts/document_loaders). For detailed documentation of all DocumentLoader features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFLoader.html).
# 
# 
# ## Overview
# ### Integration details
# 
# 
# | Class | Package | Local | Serializable | JS support|
# | :--- | :--- | :---: | :---: |  :---: |
# | [PyPDFLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ❌ | 
# ### Loader features
# | Source | Document Lazy Loading | Native Async Support
# | :---: | :---: | :---: | 
# | PyPDFLoader | ✅ | ❌ | 
# 
# ## Setup
# 
# ### Credentials
# 
# No credentials are required to use `PyPDFLoader`.

# ### Installation
# 
# To use `PyPDFLoader` you need to have the `langchain-community` python package downloaded:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain_community pypdf')


# ## Initialization
# 
# Now we can instantiate our model object and load documents:

# In[1]:


from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    "./example_data/layout-parser-paper.pdf",
)


# ## Load

# In[2]:


docs = loader.load()
docs[0]


# In[3]:


print(docs[0].metadata)


# ## Lazy Load
# 

# In[4]:


pages = []
for doc in loader.lazy_load():
    pages.append(doc)
    if len(pages) >= 10:
        # do some paged operation, e.g.
        # index.upsert(page)

        pages = []
len(pages)


# In[5]:


print(pages[0].page_content[:100])
print(pages[0].metadata)


# ## API reference
# 
# For detailed documentation of all `PyPDFLoader` features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFLoader.html
