#!/usr/bin/env python
# coding: utf-8

# # Sitemap
# 
# Extends from the `WebBaseLoader`, `SitemapLoader` loads a sitemap from a given URL, and then scrapes and loads all pages in the sitemap, returning each page as a Document.
# 
# The scraping is done concurrently. There are reasonable limits to concurrent requests, defaulting to 2 per second.  If you aren't concerned about being a good citizen, or you control the scrapped server, or don't care about load you can increase this limit. Note, while this will speed up the scraping process, it may cause the server to block you. Be careful!
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/document_loaders/web_loaders/sitemap/)|
# | :--- | :--- | :---: | :---: |  :---: |
# | [SiteMapLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.sitemap.SitemapLoader.html#langchain_community.document_loaders.sitemap.SitemapLoader) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ✅ | 
# ### Loader features
# | Source | Document Lazy Loading | Native Async Support
# | :---: | :---: | :---: | 
# | SiteMapLoader | ✅ | ❌ | 
# 
# ## Setup
# 
# To access SiteMap document loader you'll need to install the `langchain-community` integration package.
# 
# ### Credentials
# 
# No credentials are needed to run this.

# If you want to get automated best in-class tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[ ]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# Install **langchain_community**.

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-community')


# ### Fix notebook asyncio bug

# In[4]:


import nest_asyncio

nest_asyncio.apply()


# ## Initialization
# 
# Now we can instantiate our model object and load documents:

# In[ ]:


from langchain_community.document_loaders.sitemap import SitemapLoader


# In[ ]:


sitemap_loader = SitemapLoader(web_path="https://api.python.langchain.com/sitemap.xml")


# ## Load

# In[5]:


docs = sitemap_loader.load()
docs[0]


# In[6]:


print(docs[0].metadata)


# You can change the `requests_per_second` parameter to increase the max concurrent requests. and use `requests_kwargs` to pass kwargs when send requests.

# In[ ]:


sitemap_loader.requests_per_second = 2
# Optional: avoid `[SSL: CERTIFICATE_VERIFY_FAILED]` issue
sitemap_loader.requests_kwargs = {"verify": False}


# ## Lazy Load
# 
# You can also load the pages lazily in order to minimize the memory load.

# In[7]:


page = []
for doc in sitemap_loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:
        # do some paged operation, e.g.
        # index.upsert(page)

        page = []


# ## Filtering sitemap URLs
# 
# Sitemaps can be massive files, with thousands of URLs.  Often you don't need every single one of them.  You can filter the URLs by passing a list of strings or regex patterns to the `filter_urls` parameter.  Only URLs that match one of the patterns will be loaded.

# In[ ]:


loader = SitemapLoader(
    web_path="https://api.python.langchain.com/sitemap.xml",
    filter_urls=["https://api.python.langchain.com/en/latest"],
)
documents = loader.load()


# In[8]:


documents[0]


# ## Add custom scraping rules
# 
# The `SitemapLoader` uses `beautifulsoup4` for the scraping process, and it scrapes every element on the page by default. The `SitemapLoader` constructor accepts a custom scraping function. This feature can be helpful to tailor the scraping process to your specific needs; for example, you might want to avoid scraping headers or navigation elements.
# 
#  The following example shows how to develop and use a custom function to avoid navigation and header elements.

# Import the `beautifulsoup4` library and define the custom function.

# In[ ]:


pip install beautifulsoup4


# In[10]:


from bs4 import BeautifulSoup


def remove_nav_and_header_elements(content: BeautifulSoup) -> str:
    # Find all 'nav' and 'header' elements in the BeautifulSoup object
    nav_elements = content.find_all("nav")
    header_elements = content.find_all("header")

    # Remove each 'nav' and 'header' element from the BeautifulSoup object
    for element in nav_elements + header_elements:
        element.decompose()

    return str(content.get_text())


# Add your custom function to the `SitemapLoader` object.

# In[11]:


loader = SitemapLoader(
    "https://api.python.langchain.com/sitemap.xml",
    filter_urls=["https://api.python.langchain.com/en/latest/"],
    parsing_function=remove_nav_and_header_elements,
)


# ## Local Sitemap
# 
# The sitemap loader can also be used to load local files.

# In[ ]:


sitemap_loader = SitemapLoader(web_path="example_data/sitemap.xml", is_local=True)

docs = sitemap_loader.load()


# ## API reference
# 
# For detailed documentation of all SiteMapLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.sitemap.SitemapLoader.html#langchain_community.document_loaders.sitemap.SitemapLoader
