#!/usr/bin/env python
# coding: utf-8

# # SearxNG Search
#
# This notebook goes over how to use a self hosted `SearxNG` search API to search the web.
#
# You can [check this link](https://docs.searxng.org/dev/search_api.html) for more informations about `Searx API` parameters.

# In[ ]:


import pprint

from langchain_community.utilities import SearxSearchWrapper


# In[ ]:


search = SearxSearchWrapper(searx_host="http://127.0.0.1:8888")


# For some engines, if a direct `answer` is available the warpper will print the answer instead of the full list of search results. You can use the `results` method of the wrapper if you want to obtain all the results.

# In[1]:


search.run("What is the capital of France")


# ## Custom Parameters
#
# SearxNG supports [135 search engines](https://docs.searxng.org/user/configured_engines.html). You can also customize the Searx wrapper with arbitrary named parameters that will be passed to the Searx search API . In the below example we will making a more interesting use of custom search parameters from searx search api.

# In this example we will be using the `engines` parameters to query wikipedia

# In[ ]:


search = SearxSearchWrapper(
    searx_host="http://127.0.0.1:8888", k=5
)  # k is for max number of items


# In[2]:


search.run("large language model ", engines=["wiki"])


# Passing other Searx parameters for searx like `language`

# In[3]:


search = SearxSearchWrapper(searx_host="http://127.0.0.1:8888", k=1)
search.run("deep learning", language="es", engines=["wiki"])


# ## Obtaining results with metadata

# In this example we will be looking for scientific paper using the `categories` parameter and limiting the results to a `time_range` (not all engines support the time range option).
#
# We also would like to obtain the results in a structured way including metadata. For this we will be using the `results` method of the wrapper.

# In[ ]:


search = SearxSearchWrapper(searx_host="http://127.0.0.1:8888")


# In[4]:


results = search.results(
    "Large Language Model prompt",
    num_results=5,
    categories="science",
    time_range="year",
)
pprint.pp(results)


# Get papers from arxiv

# In[5]:


results = search.results(
    "Large Language Model prompt", num_results=5, engines=["arxiv"]
)
pprint.pp(results)


# In this example we query for `large language models` under the `it` category. We then filter the results that come from github.

# In[6]:


results = search.results("large language model", num_results=20, categories="it")
pprint.pp(list(filter(lambda r: r["engines"][0] == "github", results)))


# We could also directly query for results from `github` and other source forges.

# In[7]:


results = search.results(
    "large language model", num_results=20, engines=["github", "gitlab"]
)
pprint.pp(results)
