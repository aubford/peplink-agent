#!/usr/bin/env python
# coding: utf-8
---
sidebar_class_name: hidden
---
# # ChatGPT Plugins
# 
# :::warning Deprecated
# 
# OpenAI has [deprecated plugins](https://openai.com/index/chatgpt-plugins/).
# 
# :::
# 
# This example shows how to use ChatGPT Plugins within LangChain abstractions.
# 
# Note 1: This currently only works for plugins with no auth.
# 
# Note 2: There are almost certainly other ways to do this, this is just a first pass. If you have better ideas, please open a PR!

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet langchain-community')


# In[ ]:


from langchain_community.tools import AIPluginTool


# In[1]:


from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import ChatOpenAI


# In[2]:


tool = AIPluginTool.from_plugin_url("https://www.klarna.com/.well-known/ai-plugin.json")


# In[3]:


llm = ChatOpenAI(temperature=0)
tools = load_tools(["requests_all"])
tools += [tool]

agent_chain = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
agent_chain.run("what t shirts are available in klarna?")


# In[ ]:




