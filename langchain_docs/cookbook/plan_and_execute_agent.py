#!/usr/bin/env python
# coding: utf-8

# # Plan-and-execute
# 
# Plan-and-execute agents accomplish an objective by first planning what to do, then executing the sub tasks. This idea is largely inspired by [BabyAGI](https://github.com/yoheinakajima/babyagi) and then the ["Plan-and-Solve" paper](https://arxiv.org/abs/2305.04091).
# 
# The planning is almost always done by an LLM.
# 
# The execution is usually done by a separate agent (equipped with tools).

# ## Imports

# In[3]:


from langchain.chains import LLMMathChain
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import Tool
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain_openai import ChatOpenAI, OpenAI


# ## Tools

# In[4]:


search = DuckDuckGoSearchAPIWrapper()
llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
]


# ## Planner, Executor, and Agent
# 

# In[5]:


model = ChatOpenAI(temperature=0)
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor)


# ## Run example

# In[6]:


agent.run(
    "Who is the current prime minister of the UK? What is their current age raised to the 0.43 power?"
)


# In[ ]:




