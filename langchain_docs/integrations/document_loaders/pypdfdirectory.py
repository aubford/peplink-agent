#!/usr/bin/env python
# coding: utf-8

# # PyPDFDirectoryLoader
#
# This loader loads all PDF files from a specific directory.
#
# ## Overview
# ### Integration details
#
#
# | Class | Package | Local | Serializable | JS support|
# | :--- | :--- | :---: | :---: |  :---: |
# | [PyPDFDirectoryLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFDirectoryLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ❌ |
# ### Loader features
# | Source | Document Lazy Loading | Native Async Support
# | :---: | :---: | :---: |
# | PyPDFDirectoryLoader | ✅ | ❌ |
#
# ## Setup
#
# ### Credentials
#
# No credentials are needed for this loader.

# If you want to get automated best in-class tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[ ]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
#
# Install **langchain_community**.

# In[ ]:


get_ipython().run_line_magic("pip", "install -qU langchain_community")


# ## Initialization
#
# Now we can instantiate our model object and load documents:

# In[1]:


from langchain_community.document_loaders import PyPDFDirectoryLoader

directory_path = (
    "../../docs/integrations/document_loaders/example_data/layout-parser-paper.pdf"
)
loader = PyPDFDirectoryLoader("example_data/")


# ## Load

# In[2]:


docs = loader.load()
docs[0]


# In[3]:


print(docs[0].metadata)


# ## Lazy Load

# In[4]:


page = []
for doc in loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:
        # do some paged operation, e.g.
        # index.upsert(page)

        page = []


# ## API reference
#
# For detailed documentation of all PyPDFDirectoryLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFDirectoryLoader.html
