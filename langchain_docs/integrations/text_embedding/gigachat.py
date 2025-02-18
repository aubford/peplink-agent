#!/usr/bin/env python
# coding: utf-8

# # GigaChat
# This notebook shows how to use LangChain with [GigaChat embeddings](https://developers.sber.ru/portal/products/gigachat).
# To use you need to install ```gigachat``` python package.

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  langchain-gigachat")


# To get GigaChat credentials you need to [create account](https://developers.sber.ru/studio/login) and [get access to API](https://developers.sber.ru/docs/ru/gigachat/individuals-quickstart)
#
# ## Example

# In[2]:


import os
from getpass import getpass

if "GIGACHAT_CREDENTIALS" not in os.environ:
    os.environ["GIGACHAT_CREDENTIALS"] = getpass()


# In[3]:


from langchain_gigachat import GigaChatEmbeddings

embeddings = GigaChatEmbeddings(verify_ssl_certs=False, scope="GIGACHAT_API_PERS")


# In[7]:


query_result = embeddings.embed_query("The quick brown fox jumps over the lazy dog")


# In[8]:


query_result[:5]
