#!/usr/bin/env python
# coding: utf-8

# # Google Serper
# 
# This notebook goes over how to use the `Google Serper` component to search the web. First you need to sign up for a free account at [serper.dev](https://serper.dev) and get your api key.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  langchain-community')


# In[11]:


import os
import pprint

os.environ["SERPER_API_KEY"] = ""


# In[2]:


from langchain_community.utilities import GoogleSerperAPIWrapper


# In[3]:


search = GoogleSerperAPIWrapper()


# In[4]:


search.run("Obama's first name?")


# ## As part of a Self Ask With Search Chain

# In[5]:


os.environ["OPENAI_API_KEY"] = ""


# In[5]:


from langchain.agents import AgentType, initialize_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
)
self_ask_with_search.run(
    "What is the hometown of the reigning men's U.S. Open champion?"
)


# ## Obtaining results with metadata
# If you would also like to obtain the results in a structured way including metadata. For this we will be using the `results` method of the wrapper.

# In[6]:


search = GoogleSerperAPIWrapper()
results = search.results("Apple Inc.")
pprint.pp(results)


# ## Searching for Google Images
# We can also query Google Images using this wrapper. For example:

# In[7]:


search = GoogleSerperAPIWrapper(type="images")
results = search.results("Lion")
pprint.pp(results)


# ## Searching for Google News
# We can also query Google News using this wrapper. For example:

# In[8]:


search = GoogleSerperAPIWrapper(type="news")
results = search.results("Tesla Inc.")
pprint.pp(results)


# If you want to only receive news articles published in the last hour, you can do the following:

# In[9]:


search = GoogleSerperAPIWrapper(type="news", tbs="qdr:h")
results = search.results("Tesla Inc.")
pprint.pp(results)


# Some examples of the `tbs` parameter:
# 
# `qdr:h` (past hour)
# `qdr:d` (past day)
# `qdr:w` (past week)
# `qdr:m` (past month)
# `qdr:y` (past year)
# 
# You can specify intermediate time periods by adding a number:
# `qdr:h12` (past 12 hours)
# `qdr:d3` (past 3 days)
# `qdr:w2` (past 2 weeks)
# `qdr:m6` (past 6 months)
# `qdr:m2` (past 2 years)
# 
# For all supported filters simply go to [Google Search](https://google.com), search for something, click on "Tools", add your date filter and check the URL for "tbs=".
# 

# ## Searching for Google Places
# We can also query Google Places using this wrapper. For example:

# In[10]:


search = GoogleSerperAPIWrapper(type="places")
results = search.results("Italian restaurants in Upper East Side")
pprint.pp(results)

