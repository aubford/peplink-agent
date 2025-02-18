#!/usr/bin/env python
# coding: utf-8

# # Recursive URL
#
# The `RecursiveUrlLoader` lets you recursively scrape all child links from a root URL and parse them into Documents.
#
# ## Overview
# ### Integration details
#
# | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/document_loaders/web_loaders/recursive_url_loader/)|
# | :--- | :--- | :---: | :---: |  :---: |
# | [RecursiveUrlLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.recursive_url_loader.RecursiveUrlLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ✅ |
# ### Loader features
# | Source | Document Lazy Loading | Native Async Support
# | :---: | :---: | :---: |
# | RecursiveUrlLoader | ✅ | ❌ |
#

# ## Setup
#
# ### Credentials
#
# No credentials are required to use the `RecursiveUrlLoader`.
#
# ### Installation
#
# The `RecursiveUrlLoader` lives in the `langchain-community` package. There's no other required packages, though you will get richer default Document metadata if you have ``beautifulsoup4` installed as well.

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install -qU langchain-community beautifulsoup4 lxml"
)


# ## Instantiation
#
# Now we can instantiate our document loader object and load Documents:

# In[1]:


from langchain_community.document_loaders import RecursiveUrlLoader

loader = RecursiveUrlLoader(
    "https://docs.python.org/3.9/",
    # max_depth=2,
    # use_async=False,
    # extractor=None,
    # metadata_extractor=None,
    # exclude_dirs=(),
    # timeout=10,
    # check_response_status=True,
    # continue_on_failure=True,
    # prevent_outside=True,
    # base_url=None,
    # ...
)


# ## Load
#
# Use ``.load()`` to synchronously load into memory all Documents, with one
# Document per visited URL. Starting from the initial URL, we recurse through
# all linked URLs up to the specified max_depth.
#
# Let's run through a basic example of how to use the `RecursiveUrlLoader` on the [Python 3.9 Documentation](https://docs.python.org/3.9/).

# In[2]:


docs = loader.load()
docs[0].metadata


# Great! The first document looks like the root page we started from. Let's look at the metadata of the next document

# In[3]:


docs[1].metadata


# That url looks like a child of our root page, which is great! Let's move on from metadata to examine the content of one of our documents

# In[6]:


print(docs[0].page_content[:300])


# That certainly looks like HTML that comes from the url https://docs.python.org/3.9/, which is what we expected. Let's now look at some variations we can make to our basic example that can be helpful in different situations.

# ## Lazy loading
#
# If we're loading a  large number of Documents and our downstream operations can be done over subsets of all loaded Documents, we can lazily load our Documents one at a time to minimize our memory footprint:

# In[ ]:


pages = []
for doc in loader.lazy_load():
    pages.append(doc)
    if len(pages) >= 10:
        # do some paged operation, e.g.
        # index.upsert(page)

        pages = []


# In this example we never have more than 10 Documents loaded into memory at a time.

# ## Adding an Extractor
#
# By default the loader sets the raw HTML from each link as the Document page content. To parse this HTML into a more human/LLM-friendly format you can pass in a custom ``extractor`` method:

# In[21]:


import re

from bs4 import BeautifulSoup


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


loader = RecursiveUrlLoader("https://docs.python.org/3.9/", extractor=bs4_extractor)
docs = loader.load()
print(docs[0].page_content[:200])


# This looks much nicer!
#
# You can similarly pass in a `metadata_extractor` to customize how Document metadata is extracted from the HTTP response. See the [API reference](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.recursive_url_loader.RecursiveUrlLoader.html) for more on this.

# ## API reference
#
# These examples show just a few of the ways in which you can modify the default `RecursiveUrlLoader`, but there are many more modifications that can be made to best fit your use case. Using the parameters `link_regex` and `exclude_dirs` can help you filter out unwanted URLs, `aload()` and `alazy_load()` can be used for aynchronous loading, and more.
#
# For detailed information on configuring and calling the ``RecursiveUrlLoader``, please see the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.recursive_url_loader.RecursiveUrlLoader.html.
