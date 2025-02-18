#!/usr/bin/env python
# coding: utf-8

# # Docusaurus
# > [Docusaurus](https://docusaurus.io/) is a static-site generator which provides out-of-the-box documentation features.
#
# By utilizing the existing `SitemapLoader`, this loader scans and loads all pages from a given Docusaurus application and returns the main documentation content of each page as a Document.

# In[1]:


from langchain_community.document_loaders import DocusaurusLoader


# Install necessary dependencies

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet beautifulsoup4 lxml")


# In[4]:


# fixes a bug with asyncio and jupyter
import nest_asyncio

nest_asyncio.apply()


# In[5]:


loader = DocusaurusLoader("https://python.langchain.com")

docs = loader.load()


# > `SitemapLoader` also provides the ability to utilize and tweak concurrency which can help optimize the time it takes to load the source documentation. Refer to the [sitemap docs](/docs/integrations/document_loaders/sitemap) for more info.

# In[6]:


docs[0]


# ## Filtering sitemap URLs
#
# Sitemaps can contain thousands of URLs and ften you don't need every single one of them. You can filter the URLs by passing a list of strings or regex patterns to the `url_filter` parameter.  Only URLs that match one of the patterns will be loaded.

# In[14]:


loader = DocusaurusLoader(
    "https://python.langchain.com",
    filter_urls=[
        "https://python.langchain.com/docs/integrations/document_loaders/sitemap"
    ],
)
documents = loader.load()


# In[16]:


documents[0]


# ## Add custom scraping rules
#
# By default, the parser **removes** all but the main content of the docusaurus page, which is normally the `<article>` tag. You also have the option  to define an **inclusive** list HTML tags by providing them as a list utilizing the `custom_html_tags` parameter. For example:

# In[ ]:


loader = DocusaurusLoader(
    "https://python.langchain.com",
    filter_urls=[
        "https://python.langchain.com/docs/integrations/document_loaders/sitemap"
    ],
    # This will only include the content that matches these tags, otherwise they will be removed
    custom_html_tags=["#content", ".main"],
)


# You can also define an entirely custom parsing function if you need finer-grained control over the returned content for each page.
#
# The following example shows how to develop and use a custom function to avoid navigation and header elements.

# In[17]:


from bs4 import BeautifulSoup


def remove_nav_and_header_elements(content: BeautifulSoup) -> str:
    # Find all 'nav' and 'header' elements in the BeautifulSoup object
    nav_elements = content.find_all("nav")
    header_elements = content.find_all("header")

    # Remove each 'nav' and 'header' element from the BeautifulSoup object
    for element in nav_elements + header_elements:
        element.decompose()

    return str(content.get_text())


# Add your custom function to the `DocusaurusLoader` object.

# In[18]:


loader = DocusaurusLoader(
    "https://python.langchain.com",
    filter_urls=[
        "https://python.langchain.com/docs/integrations/document_loaders/sitemap"
    ],
    parsing_function=remove_nav_and_header_elements,
)
