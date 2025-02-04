#!/usr/bin/env python
# coding: utf-8

# # Google Jobs
# 
# This notebook goes over how to use the Google Jobs Tool to fetch current Job postings.
# 
# First, you need to sign up for an `SerpApi key` key at: https://serpapi.com/users/sign_up.
# 
# Then you must install `google-search-results` with the command:
#     `pip install google-search-results`
# 
# Then you will need to set the environment variable `SERPAPI_API_KEY` to your `SerpApi key`
# 
# If you don't have one you can register a free account on https://serpapi.com/users/sign_up and get your api key here: https://serpapi.com/manage-api-key
# 

# If you are using conda environment, you can set up using the following commands in kernal:
conda activate [your env name]
conda env confiv vars SERPAPI_API_KEY='[your serp api key]'
# ## Use the Tool

# In[1]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  google-search-results langchain-community')


# In[1]:


import os

from langchain_community.tools.google_jobs import GoogleJobsQueryRun
from langchain_community.utilities.google_jobs import GoogleJobsAPIWrapper

os.environ["SERPAPI_API_KEY"] = "[your serpapi key]"
tool = GoogleJobsQueryRun(api_wrapper=GoogleJobsAPIWrapper())


# In[2]:


tool.run("Can I get an entry level job posting related to physics")


# # use it with langchain

# In[6]:


import os

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import OpenAI

OpenAI.api_key = os.environ["OPENAI_API_KEY"]
llm = OpenAI()
tools = load_tools(["google-jobs"], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
agent.run("give me an entry level job posting related to physics")


# In[ ]:




