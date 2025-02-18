#!/usr/bin/env python
# coding: utf-8

# # URL
#
# This example covers how to load `HTML` documents from a list of `URLs` into the `Document` format that we can use downstream.
#
# ## Unstructured URL Loader
#
# For the examples below, please install the `unstructured` library and see [this guide](/docs/integrations/providers/unstructured/) for more instructions on setting up Unstructured locally, including setting up required system dependencies:

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet unstructured")


# In[2]:


from langchain_community.document_loaders import UnstructuredURLLoader

urls = [
    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-8-2023",
    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-9-2023",
]


# Pass in ssl_verify=False with headers=headers to get past ssl_verification errors.

# In[3]:


loader = UnstructuredURLLoader(urls=urls)

data = loader.load()

data[0]


# ## Selenium URL Loader
#
# This covers how to load HTML documents from a list of URLs using the `SeleniumURLLoader`.
#
# Using `Selenium` allows us to load pages that require JavaScript to render.
#
#
# To use the `SeleniumURLLoader`, you have to install `selenium` and `unstructured`.

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet selenium unstructured")


# In[6]:


from langchain_community.document_loaders import SeleniumURLLoader

urls = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://goo.gl/maps/NDSHwePEyaHMFGwh8",
]

loader = SeleniumURLLoader(urls=urls)

data = loader.load()

data[1]


# ## Playwright URL Loader
#
# >[Playwright](https://github.com/microsoft/playwright) is an open-source automation tool developed by `Microsoft` that allows you to programmatically control and automate web browsers. It is designed for end-to-end testing, scraping, and automating tasks across various web browsers such as `Chromium`, `Firefox`, and `WebKit`.
#
# This covers how to load HTML documents from a list of URLs using the `PlaywrightURLLoader`.
#
# [Playwright](https://playwright.dev/) enables reliable end-to-end testing for modern web apps.
#
# As in the Selenium case, `Playwright` allows us to load and render the JavaScript pages.
#
# To use the `PlaywrightURLLoader`, you have to install `playwright` and `unstructured`. Additionally, you have to install the `Playwright Chromium` browser:

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet playwright unstructured")


# In[9]:


get_ipython().system("playwright install")


# Currently, nly the async method supported:

# In[14]:


from langchain_community.document_loaders import PlaywrightURLLoader

urls = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://goo.gl/maps/NDSHwePEyaHMFGwh8",
]

loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])

data = await loader.aload()

data[0]
