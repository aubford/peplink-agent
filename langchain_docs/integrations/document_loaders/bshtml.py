#!/usr/bin/env python
# coding: utf-8

# # BSHTMLLoader
# 
# 
# This notebook provides a quick overview for getting started with BeautifulSoup4 [document loader](https://python.langchain.com/docs/concepts/document_loaders). For detailed documentation of all __ModuleName__Loader features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.html_bs.BSHTMLLoader.html).
# 
# 
# ## Overview
# ### Integration details
# 
# 
# | Class | Package | Local | Serializable | JS support|
# | :--- | :--- | :---: | :---: |  :---: |
# | [BSHTMLLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.html_bs.BSHTMLLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ❌ | 
# ### Loader features
# | Source | Document Lazy Loading | Native Async Support
# | :---: | :---: | :---: | 
# | BSHTMLLoader | ✅ | ❌ | 
# 
# ## Setup
# 
# To access BSHTMLLoader document loader you'll need to install the `langchain-community` integration package and the `bs4` python package.
# 
# ### Credentials
# 
# No credentials are needed to use the `BSHTMLLoader` class.

# If you want to get automated best in-class tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[ ]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# Install **langchain_community** and **bs4**.

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain_community bs4')


# ## Initialization
# 
# Now we can instantiate our model object and load documents:
# 
# - TODO: Update model instantiation with relevant params.

# In[2]:


from langchain_community.document_loaders import BSHTMLLoader

loader = BSHTMLLoader(
    file_path="./example_data/fake-content.html",
)


# ## Load

# In[3]:


docs = loader.load()
docs[0]


# In[4]:


print(docs[0].metadata)


# ## Lazy Load

# In[9]:


page = []
for doc in loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:
        # do some paged operation, e.g.
        # index.upsert(page)

        page = []
page[0]


# ## Adding separator to BS4
# 
# We can also pass a separator to use when calling get_text on the soup

# In[13]:


loader = BSHTMLLoader(
    file_path="./example_data/fake-content.html", get_text_separator=", "
)

docs = loader.load()
print(docs[0])


# ## API reference
# 
# For detailed documentation of all BSHTMLLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.html_bs.BSHTMLLoader.html
