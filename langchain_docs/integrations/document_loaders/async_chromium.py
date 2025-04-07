#!/usr/bin/env python
# coding: utf-8

# # Async Chromium
# 
# Chromium is one of the browsers supported by Playwright, a library used to control browser automation. 
# 
# By running `p.chromium.launch(headless=True)`, we are launching a headless instance of Chromium. 
# 
# Headless mode means that the browser is running without a graphical user interface.
# 
# In the below example we'll use the `AsyncChromiumLoader` to loads the page, and then the [`Html2TextTransformer`](/docs/integrations/document_transformers/html2text/) to strip out the HTML tags and other semantic information.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet playwright beautifulsoup4 html2text')
get_ipython().system('playwright install')


# **Note:** If you are using Jupyter notebooks, you might also need to install and apply `nest_asyncio` before loading the documents like this:

# In[ ]:


get_ipython().system('pip install nest-asyncio')
import nest_asyncio

nest_asyncio.apply()


# In[5]:


from langchain_community.document_loaders import AsyncChromiumLoader

urls = ["https://docs.smith.langchain.com/"]
loader = AsyncChromiumLoader(urls, user_agent="MyAppUserAgent")
docs = loader.load()
docs[0].page_content[0:100]


# Now let's transform the documents into a more readable syntax using the transformer:

# In[6]:


from langchain_community.document_transformers import Html2TextTransformer

html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)
docs_transformed[0].page_content[0:500]

