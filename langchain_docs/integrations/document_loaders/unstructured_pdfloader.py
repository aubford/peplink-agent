#!/usr/bin/env python
# coding: utf-8

# # UnstructuredPDFLoader
#
# ## Overview
#
# [Unstructured](https://unstructured-io.github.io/unstructured/) supports a common interface for working with unstructured or semi-structured file formats, such as Markdown or PDF. LangChain's [UnstructuredPDFLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.UnstructuredPDFLoader.html) integrates with Unstructured to parse PDF documents into LangChain [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) objects.
#
# Please see [this page](/docs/integrations/providers/unstructured/) for more information on installing system requirements.
#
#
# ### Integration details
#
#
# | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/document_loaders/file_loaders/unstructured/)|
# | :--- | :--- | :---: | :---: |  :---: |
# | [UnstructuredPDFLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.UnstructuredPDFLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ✅ |
# ### Loader features
# | Source | Document Lazy Loading | Native Async Support
# | :---: | :---: | :---: |
# | UnstructuredPDFLoader | ✅ | ❌ |
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
# Install **langchain_community** and **unstructured**.

# In[ ]:


get_ipython().run_line_magic("pip", "install -qU langchain-community unstructured")


# ## Initialization
#
# Now we can initialize our loader:

# In[3]:


from langchain_community.document_loaders import UnstructuredPDFLoader

file_path = "./example_data/layout-parser-paper.pdf"
loader = UnstructuredPDFLoader(file_path)


# ## Load

# In[4]:


docs = loader.load()
docs[0]


# In[5]:


print(docs[0].metadata)


# ### Retain Elements
#
# Under the hood, Unstructured creates different "elements" for different chunks of text. By default we combine those together, but you can easily keep that separation by specifying `mode="elements"`.

# In[6]:


file_path = "./example_data/layout-parser-paper.pdf"
loader = UnstructuredPDFLoader(file_path, mode="elements")

data = loader.load()
data[0]


# See the full set of element types for this particular document:

# In[7]:


set(doc.metadata["category"] for doc in data)


# ### Fetching remote PDFs using Unstructured
#
# This covers how to load online PDFs into a document format that we can use downstream. This can be used for various online PDF sites such as https://open.umn.edu/opentextbooks/textbooks/ and https://arxiv.org/archive/
#
# Note: all other PDF loaders can also be used to fetch remote PDFs, but `OnlinePDFLoader` is a legacy function, and works specifically with `UnstructuredPDFLoader`.

# In[8]:


from langchain_community.document_loaders import OnlinePDFLoader

loader = OnlinePDFLoader("https://arxiv.org/pdf/2302.03803.pdf")
data = loader.load()
data[0]


# ## Lazy Load

# In[9]:


page = []
for doc in loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:
        # do some paged operation, e.g.
        # index.upsert(page)

        page = []


# ## API reference
#
# For detailed documentation of all UnstructuredPDFLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.UnstructuredPDFLoader.html
