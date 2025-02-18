#!/usr/bin/env python
# coding: utf-8

# # How to cache chat model responses
#
# :::info Prerequisites
#
# This guide assumes familiarity with the following concepts:
# - [Chat models](/docs/concepts/chat_models)
# - [LLMs](/docs/concepts/text_llms)
#
# :::
#
# LangChain provides an optional caching layer for [chat models](/docs/concepts/chat_models). This is useful for two main reasons:
#
# - It can save you money by reducing the number of API calls you make to the LLM provider, if you're often requesting the same completion multiple times. This is especially useful during app development.
# - It can speed up your application by reducing the number of API calls you make to the LLM provider.
#
# This guide will walk you through how to enable this in your apps.

# import ChatModelTabs from "@theme/ChatModelTabs";
#
# <ChatModelTabs customVarName="llm" />
#

# In[1]:


# | output: false
# | echo: false

import os
from getpass import getpass

from langchain_openai import ChatOpenAI

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass()

llm = ChatOpenAI()


# In[2]:


# <!-- ruff: noqa: F821 -->
from langchain_core.globals import set_llm_cache


# ## In Memory Cache
#
# This is an ephemeral cache that stores model calls in memory. It will be wiped when your environment restarts, and is not shared across processes.

# In[3]:


get_ipython().run_cell_magic(
    "time",
    "",
    'from langchain_core.caches import InMemoryCache\n\nset_llm_cache(InMemoryCache())\n\n# The first time, it is not yet in cache, so it should take longer\nllm.invoke("Tell me a joke")\n',
)


# In[4]:


get_ipython().run_cell_magic(
    "time",
    "",
    '# The second time it is, so it goes faster\nllm.invoke("Tell me a joke")\n',
)


# ## SQLite Cache
#
# This cache implementation uses a `SQLite` database to store responses, and will last across process restarts.

# In[5]:


get_ipython().system("rm .langchain.db")


# In[6]:


# We can do the same thing with a SQLite cache
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))


# In[7]:


get_ipython().run_cell_magic(
    "time",
    "",
    '# The first time, it is not yet in cache, so it should take longer\nllm.invoke("Tell me a joke")\n',
)


# In[8]:


get_ipython().run_cell_magic(
    "time",
    "",
    '# The second time it is, so it goes faster\nllm.invoke("Tell me a joke")\n',
)


# ## Next steps
#
# You've now learned how to cache model responses to save time and money.
#
# Next, check out the other how-to guides chat models in this section, like [how to get a model to return structured output](/docs/how_to/structured_output) or [how to create your own custom chat model](/docs/how_to/custom_chat_model).
