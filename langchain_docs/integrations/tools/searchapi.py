#!/usr/bin/env python
# coding: utf-8

# # SearchApi
# 
# This notebook shows examples of how to use SearchApi to search the web. Go to [https://www.searchapi.io/](https://www.searchapi.io/) to sign up for a free account and get API key.

# In[12]:


import os

os.environ["SEARCHAPI_API_KEY"] = ""


# In[2]:


from langchain_community.utilities import SearchApiAPIWrapper


# In[3]:


search = SearchApiAPIWrapper()


# In[4]:


search.run("Obama's first name?")


# ## Using as part of a Self Ask With Search Chain

# In[13]:


os.environ["OPENAI_API_KEY"] = ""


# In[7]:


from langchain.agents import AgentType, initialize_agent
from langchain_community.utilities import SearchApiAPIWrapper
from langchain_core.tools import Tool
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
search = SearchApiAPIWrapper()
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
self_ask_with_search.run("Who lived longer: Plato, Socrates, or Aristotle?")


# ## Custom parameters
# 
# SearchApi wrapper can be customized to use different engines like [Google News](https://www.searchapi.io/docs/google-news), [Google Jobs](https://www.searchapi.io/docs/google-jobs), [Google Scholar](https://www.searchapi.io/docs/google-scholar), or others which can be found in [SearchApi](https://www.searchapi.io/docs/google) documentation. All parameters supported by SearchApi can be passed when executing the query. 

# In[8]:


search = SearchApiAPIWrapper(engine="google_jobs")


# In[9]:


search.run("AI Engineer", location="Portugal", gl="pt")[0:500]


# ## Getting results with metadata

# In[10]:


import pprint


# In[11]:


search = SearchApiAPIWrapper(engine="google_scholar")
results = search.results("Large Language Models")
pprint.pp(results)

