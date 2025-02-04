#!/usr/bin/env python
# coding: utf-8

# # SerpAPI
# 
# This notebook goes over how to use the SerpAPI component to search the web.

# In[1]:


from langchain_community.utilities import SerpAPIWrapper


# In[2]:


search = SerpAPIWrapper()


# In[3]:


search.run("Obama's first name?")


# ## Custom Parameters
# You can also customize the SerpAPI wrapper with arbitrary parameters. For example, in the below example we will use `bing` instead of `google`.

# In[2]:


params = {
    "engine": "bing",
    "gl": "us",
    "hl": "en",
}
search = SerpAPIWrapper(params=params)


# In[3]:


search.run("Obama's first name?")


# In[ ]:


from langchain_core.tools import Tool

# You can create the tool to pass to an agent
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=search.run,
)

