#!/usr/bin/env python
# coding: utf-8

# # WebBaseLoader
#
# This covers how to use `WebBaseLoader` to load all text from `HTML` webpages into a document format that we can use downstream. For more custom logic for loading webpages look at some child class examples such as `IMSDbLoader`, `AZLyricsLoader`, and `CollegeConfidentialLoader`.
#
# If you don't want to worry about website crawling, bypassing JS-blocking sites, and data cleaning, consider using `FireCrawlLoader` or the faster option `SpiderLoader`.
#
# ## Overview
# ### Integration details
#
# - TODO: Fill in table features.
# - TODO: Remove JS support link if not relevant, otherwise ensure link is correct.
# - TODO: Make sure API reference links are correct.
#
# | Class | Package | Local | Serializable | JS support|
# | :--- | :--- | :---: | :---: |  :---: |
# | [WebBaseLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ❌ |
# ### Loader features
# | Source | Document Lazy Loading | Native Async Support
# | :---: | :---: | :---: |
# | WebBaseLoader | ✅ | ✅ |
#
# ## Setup
#
# ### Credentials
#
# `WebBaseLoader` does not require any credentials.
#
# ### Installation
#
# To use the `WebBaseLoader` you first need to install the `langchain-community` python package.
#

# In[ ]:


get_ipython().run_line_magic("pip", "install -qU langchain_community beautifulsoup4")


# ## Initialization
#
# Now we can instantiate our model object and load documents:

# In[2]:


from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.example.com/")


# To bypass SSL verification errors during fetching, you can set the "verify" option:
#
# `loader.requests_kwargs = {'verify':False}`
#
# ### Initialization with multiple pages
#
# You can also pass in a list of pages to load from.

# In[3]:


loader_multiple_pages = WebBaseLoader(
    ["https://www.example.com/", "https://google.com"]
)


# ## Load

# In[4]:


docs = loader.load()

docs[0]


# In[5]:


print(docs[0].metadata)


# ### Load multiple urls concurrently
#
# You can speed up the scraping process by scraping and parsing multiple urls concurrently.
#
# There are reasonable limits to concurrent requests, defaulting to 2 per second.  If you aren't concerned about being a good citizen, or you control the server you are scraping and don't care about load, you can change the `requests_per_second` parameter to increase the max concurrent requests.  Note, while this will speed up the scraping process, but may cause the server to block you.  Be careful!

# In[6]:


get_ipython().run_line_magic("pip", "install -qU  nest_asyncio")

# fixes a bug with asyncio and jupyter
import nest_asyncio

nest_asyncio.apply()


# In[8]:


loader = WebBaseLoader(["https://www.example.com/", "https://google.com"])
loader.requests_per_second = 1
docs = loader.aload()
docs


# ### Loading a xml file, or using a different BeautifulSoup parser
#
# You can also look at `SitemapLoader` for an example of how to load a sitemap file, which is an example of using this feature.

# In[9]:


loader = WebBaseLoader(
    "https://www.govinfo.gov/content/pkg/CFR-2018-title10-vol3/xml/CFR-2018-title10-vol3-sec431-86.xml"
)
loader.default_parser = "xml"
docs = loader.load()
docs


# ## Lazy Load
#
# You can use lazy loading to only load one page at a time in order to minimize memory requirements.

# In[10]:


pages = []
for doc in loader.lazy_load():
    pages.append(doc)

print(pages[0].page_content[:100])
print(pages[0].metadata)


# ### Async

# In[12]:


pages = []
async for doc in loader.alazy_load():
    pages.append(doc)

print(pages[0].page_content[:100])
print(pages[0].metadata)


# ## Using proxies
#
# Sometimes you might need to use proxies to get around IP blocks. You can pass in a dictionary of proxies to the loader (and `requests` underneath) to use them.

# In[ ]:


loader = WebBaseLoader(
    "https://www.walmart.com/search?q=parrots",
    proxies={
        "http": "http://{username}:{password}:@proxy.service.com:6666/",
        "https": "https://{username}:{password}:@proxy.service.com:6666/",
    },
)
docs = loader.load()


# ## API reference
#
# For detailed documentation of all `WebBaseLoader` features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html
