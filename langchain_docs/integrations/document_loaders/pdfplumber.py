#!/usr/bin/env python
# coding: utf-8

# # PDFPlumber
#
# Like PyMuPDF, the output Documents contain detailed metadata about the PDF and its pages, and returns one document per page.
#
# ## Overview
# ### Integration details
#
# | Class | Package | Local | Serializable | JS support|
# | :--- | :--- | :---: | :---: |  :---: |
# | [PDFPlumberLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PDFPlumberLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ❌ |
# ### Loader features
# | Source | Document Lazy Loading | Native Async Support
# | :---: | :---: | :---: |
# | PDFPlumberLoader | ✅ | ❌ |
#
# ## Setup
#
# ### Credentials
#
# No credentials are needed to use this loader.

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

# In[4]:


from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader("./example_data/layout-parser-paper.pdf")


# ## Load

# In[5]:


docs = loader.load()
docs[0]


# In[6]:


print(docs[0].metadata)


# ## Lazy Load

# In[7]:


page = []
for doc in loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:
        # do some paged operation, e.g.
        # index.upsert(page)

        page = []


# ## API reference
#
# For detailed documentation of all PDFPlumberLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PDFPlumberLoader.html
