#!/usr/bin/env python
# coding: utf-8

# # UnstructuredXMLLoader
# 
# This notebook provides a quick overview for getting started with UnstructuredXMLLoader [document loader](https://python.langchain.com/docs/concepts/document_loaders). The `UnstructuredXMLLoader` is used to load `XML` files. The loader works with `.xml` files. The page content will be the text extracted from the XML tags.
# 
# 
# ## Overview
# ### Integration details
# 
# 
# | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/document_loaders/file_loaders/unstructured/)|
# | :--- | :--- | :---: | :---: |  :---: |
# | [UnstructuredXMLLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.xml.UnstructuredXMLLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ✅ | 
# ### Loader features
# | Source | Document Lazy Loading | Native Async Support
# | :---: | :---: | :---: | 
# | UnstructuredXMLLoader | ✅ | ❌ | 
# 
# ## Setup
# 
# To access UnstructuredXMLLoader document loader you'll need to install the `langchain-community` integration package.
# 
# ### Credentials
# 
# No credentials are needed to use the UnstructuredXMLLoader

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

# In[2]:


from langchain_community.document_loaders import UnstructuredXMLLoader

loader = UnstructuredXMLLoader(
    "./example_data/factbook.xml",
)


# ## Load

# In[3]:


docs = loader.load()
docs[0]


# In[4]:


print(docs[0].metadata)


# ## Lazy Load

# In[5]:


page = []
for doc in loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:
        # do some paged operation, e.g.
        # index.upsert(page)

        page = []


# ## API reference
# 
# For detailed documentation of all __ModuleName__Loader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.xml.UnstructuredXMLLoader.html
