#!/usr/bin/env python
# coding: utf-8

# # Yahoo Finance News
# 
# This notebook goes over how to use the `yahoo_finance_news` tool with an agent. 
# 
# 
# ## Setting up
# 
# First, you need to install `yfinance` python package.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  yfinance')


# ## Example with Chain

# In[4]:


import os

os.environ["OPENAI_API_KEY"] = "..."


# In[26]:


from langchain.agents import AgentType, initialize_agent
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0.0)
tools = [YahooFinanceNewsTool()]
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


# In[19]:


agent_chain.invoke(
    "What happened today with Microsoft stocks?",
)


# In[20]:


agent_chain.invoke(
    "How does Microsoft feels today comparing with Nvidia?",
)


# # How YahooFinanceNewsTool works?

# In[37]:


tool = YahooFinanceNewsTool()


# In[38]:


tool.invoke("NVDA")


# In[40]:


res = tool.invoke("AAPL")
print(res)


# In[ ]:




