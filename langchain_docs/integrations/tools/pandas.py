#!/usr/bin/env python
# coding: utf-8

# # Pandas Dataframe
# 
# This notebook shows how to use agents to interact with a `Pandas DataFrame`. It is mostly optimized for question answering.
# 
# **NOTE: this agent calls the `Python` agent under the hood, which executes LLM generated Python code - this can be bad if the LLM generated Python code is harmful. Use cautiously.**
# 
# **NOTE: Since langchain migrated to v0.3 you should upgrade langchain_openai and langchain.   This would avoid import errors.**
# 
# 
# pip install --upgrade langchain_openai
# pip install --upgrade langchain
# 

# In[1]:


from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI


# In[2]:


import pandas as pd
from langchain_openai import OpenAI

df = pd.read_csv(
    "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
)


# ## Using `ZERO_SHOT_REACT_DESCRIPTION`
# 
# This shows how to initialize the agent using the `ZERO_SHOT_REACT_DESCRIPTION` agent type.

# In[3]:


agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)


# ## Using OpenAI Functions
# 
# This shows how to initialize the agent using the OPENAI_FUNCTIONS agent type. Note that this is an alternative to the above.

# In[4]:


agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)


# In[5]:


agent.invoke("how many rows are there?")


# In[5]:


agent.invoke("how many people have more than 3 siblings")


# In[6]:


agent.invoke("whats the square root of the average age?")


# ## Multi DataFrame Example
# 
# This next part shows how the agent can interact with multiple dataframes passed in as a list.

# In[7]:


df1 = df.copy()
df1["Age"] = df1["Age"].fillna(df1["Age"].mean())


# In[8]:


agent = create_pandas_dataframe_agent(OpenAI(temperature=0), [df, df1], verbose=True)
agent.invoke("how many rows in the age column are different?")


# In[ ]:




