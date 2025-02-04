#!/usr/bin/env python
# coding: utf-8

# # Google Search
# 
# This notebook goes over how to use the google search component.
# 
# First, you need to set up the proper API keys and environment variables. To set it up, create the GOOGLE_API_KEY in the Google Cloud credential console (https://console.cloud.google.com/apis/credentials) and a GOOGLE_CSE_ID using the Programmable Search Engine (https://programmablesearchengine.google.com/controlpanel/create). Next, it is good to follow the instructions found [here](https://stackoverflow.com/questions/37083058/programmatically-searching-google-in-python-using-custom-search).
# 
# Then we will need to set some environment variables.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  langchain-google-community')


# In[1]:


import os

os.environ["GOOGLE_CSE_ID"] = ""
os.environ["GOOGLE_API_KEY"] = ""


# In[2]:


from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper()

tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)


# In[3]:


tool.run("Obama's first name?")


# ## Number of Results
# You can use the `k` parameter to set the number of results

# In[4]:


search = GoogleSearchAPIWrapper(k=1)

tool = Tool(
    name="I'm Feeling Lucky",
    description="Search Google and return the first result.",
    func=search.run,
)


# In[5]:


tool.run("python")


# 'The official home of the Python Programming Language.'

# ## Metadata Results

# Run query through GoogleSearch and return snippet, title, and link metadata.
# 
# - Snippet: The description of the result.
# - Title: The title of the result.
# - Link: The link to the result.

# In[6]:


search = GoogleSearchAPIWrapper()


def top5_results(query):
    return search.results(query, 5)


tool = Tool(
    name="Google Search Snippets",
    description="Search Google for recent results.",
    func=top5_results,
)


# In[ ]:




