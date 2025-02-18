#!/usr/bin/env python
# coding: utf-8

# # Beautiful Soup
#
# >[Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) is a Python package for parsing
# > HTML and XML documents (including having malformed markup, i.e. non-closed tags, so named after tag soup).
# > It creates a parse tree for parsed pages that can be used to extract data from HTML,[3] which
# > is useful for web scraping.
#
# `Beautiful Soup` offers fine-grained control over HTML content, enabling specific tag extraction, removal, and content cleaning.
#
# It's suited for cases where you want to extract specific information and clean up the HTML content according to your needs.
#
# For example, we can scrape text content within `<p>, <li>, <div>, and <a>` tags from the HTML content:
#
# * `<p>`: The paragraph tag. It defines a paragraph in HTML and is used to group together related sentences and/or phrases.
#
# * `<li>`: The list item tag. It is used within ordered (`<ol>`) and unordered (`<ul>`) lists to define individual items within the list.
#
# * `<div>`: The division tag. It is a block-level element used to group other inline or block-level elements.
#
# * `<a>`: The anchor tag. It is used to define hyperlinks.

# In[2]:


from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

# Load HTML
loader = AsyncChromiumLoader(["https://www.wsj.com"])
html = loader.load()


# In[3]:


# Transform
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(
    html, tags_to_extract=["p", "li", "div", "a"]
)


# In[4]:


docs_transformed[0].page_content[0:500]
