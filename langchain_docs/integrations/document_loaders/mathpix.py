#!/usr/bin/env python
# coding: utf-8

# # MathPixPDFLoader
#
# Inspired by Daniel Gross's snippet here: [https://gist.github.com/danielgross/3ab4104e14faccc12b49200843adab21](https://gist.github.com/danielgross/3ab4104e14faccc12b49200843adab21)
#
# ## Overview
# ### Integration details
#
# | Class | Package | Local | Serializable | JS support|
# | :--- | :--- | :---: | :---: |  :---: |
# | [MathPixPDFLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.MathpixPDFLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ❌ |
# ### Loader features
# | Source | Document Lazy Loading | Native Async Support
# | :---: | :---: | :---: |
# | MathPixPDFLoader | ✅ | ❌ |
#
# ## Setup
#
# ### Credentials
#
# Sign up for Mathpix and [create an API key](https://mathpix.com/docs/ocr/creating-an-api-key) to set the `MATHPIX_API_KEY` variables in your environment

# In[ ]:


import getpass
import os

if "MATHPIX_API_KEY" not in os.environ:
    os.environ["MATHPIX_API_KEY"] = getpass.getpass("Enter your Mathpix API key: ")


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
# Now we are ready to initialize our loader:

# In[ ]:


from langchain_community.document_loaders import MathpixPDFLoader

file_path = "./example_data/layout-parser-paper.pdf"
loader = MathpixPDFLoader(file_path)


# ## Load

# In[ ]:


docs = loader.load()
docs[0]


# In[ ]:


print(docs[0].metadata)


# ## Lazy Load

# In[ ]:


page = []
for doc in loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:
        # do some paged operation, e.g.
        # index.upsert(page)

        page = []


# ## API reference
#
# For detailed documentation of all MathpixPDFLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.MathpixPDFLoader.html
