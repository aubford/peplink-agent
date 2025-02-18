#!/usr/bin/env python
# coding: utf-8

# # iFixit
#
# >[iFixit](https://www.ifixit.com) is the largest, open repair community on the web. The site contains nearly 100k repair manuals, 200k Questions & Answers on 42k devices, and all the data is licensed under CC-BY-NC-SA 3.0.
#
# This loader will allow you to download the text of a repair guide, text of Q&A's and wikis from devices on `iFixit` using their open APIs.  It's incredibly useful for context related to technical documents and answers to questions about devices in the corpus of data on `iFixit`.

# In[1]:


from langchain_community.document_loaders import IFixitLoader


# In[2]:


loader = IFixitLoader("https://www.ifixit.com/Teardown/Banana+Teardown/811")
data = loader.load()


# In[3]:


data


# In[4]:


loader = IFixitLoader(
    "https://www.ifixit.com/Answers/View/318583/My+iPhone+6+is+typing+and+opening+apps+by+itself"
)
data = loader.load()


# In[5]:


data


# In[7]:


loader = IFixitLoader("https://www.ifixit.com/Device/Standard_iPad")
data = loader.load()


# In[8]:


data


# ## Searching iFixit using /suggest
#
# If you're looking for a more general way to search iFixit based on a keyword or phrase, the /suggest endpoint will return content related to the search term, then the loader will load the content from each of the suggested items and prep and return the documents.

# In[2]:


data = IFixitLoader.load_suggestions("Banana")


# In[3]:


data
