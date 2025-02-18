#!/usr/bin/env python
# coding: utf-8

# # Browserless
#
# Browserless is a service that allows you to run headless Chrome instances in the cloud. It's a great way to run browser-based automation at scale without having to worry about managing your own infrastructure.
#
# To use Browserless as a document loader, initialize a `BrowserlessLoader` instance as shown in this notebook. Note that by default, `BrowserlessLoader` returns the `innerText` of the page's `body` element. To disable this and get the raw HTML, set `text_content` to `False`.

# In[11]:


from langchain_community.document_loaders import BrowserlessLoader


# In[12]:


BROWSERLESS_API_TOKEN = "YOUR_BROWSERLESS_API_TOKEN"


# In[14]:


loader = BrowserlessLoader(
    api_token=BROWSERLESS_API_TOKEN,
    urls=[
        "https://en.wikipedia.org/wiki/Document_classification",
    ],
    text_content=True,
)

documents = loader.load()

print(documents[0].page_content[:1000])
