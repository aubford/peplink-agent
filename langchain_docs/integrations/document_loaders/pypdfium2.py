#!/usr/bin/env python
# coding: utf-8

# # PyPDFium2Loader
# 
# 
# This notebook provides a quick overview for getting started with PyPDFium2 [document loader](https://python.langchain.com/docs/concepts/document_loaders). For detailed documentation of all __ModuleName__Loader features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFium2Loader.html).
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | JS support|
# | :--- | :--- | :---: | :---: |  :---: |
# | [PyPDFium2Loader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFium2Loader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ❌ | 
# ### Loader features
# | Source | Document Lazy Loading | Native Async Support
# | :---: | :---: | :---: | 
# | PyPDFium2Loader | ✅ | ❌ | 
# 
# ## Setup
# 
# 
# To access PyPDFium2 document loader you'll need to install the `langchain-community` integration package.
# 
# ### Credentials
# 
# No credentials are needed.

# If you want to get automated best in-class tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[ ]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# Install **langchain_community**.

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain_community')


# ## Initialization
# 
# Now we can instantiate our model object and load documents:

# In[3]:


from langchain_community.document_loaders import PyPDFium2Loader

file_path = "./example_data/layout-parser-paper.pdf"
loader = PyPDFium2Loader(file_path)


# ## Load

# In[4]:


docs = loader.load()
docs[0]


# In[5]:


print(docs[0].metadata)


# ## Lazy Load

# In[6]:


page = []
for doc in loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:
        # do some paged operation, e.g.
        # index.upsert(page)

        page = []


# ## API reference
# 
# For detailed documentation of all PyPDFium2Loader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFium2Loader.html
