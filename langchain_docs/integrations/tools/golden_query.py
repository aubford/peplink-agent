#!/usr/bin/env python
# coding: utf-8

# # Golden Query
# 
# >[Golden](https://golden.com) provides a set of natural language APIs for querying and enrichment using the Golden Knowledge Graph e.g. queries such as: `Products from OpenAI`, `Generative ai companies with series a funding`, and `rappers who invest` can be used to retrieve structured data about relevant entities.
# >
# >The `golden-query` langchain tool is a wrapper on top of the [Golden Query API](https://docs.golden.com/reference/query-api) which enables programmatic access to these results.
# >See the [Golden Query API docs](https://docs.golden.com/reference/query-api) for more information.
# 
# 
# This notebook goes over how to use the `golden-query` tool.
# 
# - Go to the [Golden API docs](https://docs.golden.com/) to get an overview about the Golden API.
# - Get your API key from the [Golden API Settings](https://golden.com/settings/api) page.
# - Save your API key into GOLDEN_API_KEY env variable

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-community')


# In[ ]:


import os

os.environ["GOLDEN_API_KEY"] = ""


# In[ ]:


from langchain_community.utilities.golden_query import GoldenQueryAPIWrapper


# In[ ]:


golden_query = GoldenQueryAPIWrapper()


# In[ ]:


import json

json.loads(golden_query.run("companies in nanotech"))

