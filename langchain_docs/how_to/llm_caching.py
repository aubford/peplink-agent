#!/usr/bin/env python
# coding: utf-8

# # How to cache LLM responses
#
# LangChain provides an optional [caching](/docs/concepts/chat_models/#caching) layer for LLMs. This is useful for two reasons:
#
# It can save you money by reducing the number of API calls you make to the LLM provider, if you're often requesting the same completion multiple times.
# It can speed up your application by reducing the number of API calls you make to the LLM provider.
#

# In[ ]:


get_ipython().run_line_magic("pip", "install -qU langchain_openai langchain_community")

import os
from getpass import getpass

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass()
# Please manually enter OpenAI Key


# In[2]:


from langchain_core.globals import set_llm_cache
from langchain_openai import OpenAI

# To make the caching really obvious, lets use a slower and older model.
# Caching supports newer chat models as well.
llm = OpenAI(model="gpt-3.5-turbo-instruct", n=2, best_of=2)


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


# In[ ]:
