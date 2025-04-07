#!/usr/bin/env python
# coding: utf-8

# # Bearly Code Interpreter
# 
# > Bearly Code Interpreter allows for remote execution of code. This makes it perfect for a code sandbox for agents, to allow for safe implementation of things like Code Interpreter
# 
# Get your api key here: https://bearly.ai/dashboard/developers

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet langchain-community')


# In this notebook, we will create an example of an agent that uses Bearly to interact with data

# In[ ]:


from langchain_community.tools import BearlyInterpreterTool


# In[9]:


from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI


# Initialize the interpreter

# In[2]:


bearly_tool = BearlyInterpreterTool(api_key="...")


# Let's add some files to the sandbox

# In[4]:


bearly_tool.add_file(
    source_path="sample_data/Bristol.pdf", target_path="Bristol.pdf", description=""
)
bearly_tool.add_file(
    source_path="sample_data/US_GDP.csv", target_path="US_GDP.csv", description=""
)


# Create a `Tool` object now. This is necessary, because we added the files, and we want the tool description to reflect that

# In[5]:


tools = [bearly_tool.as_tool()]


# In[6]:


tools[0].name


# In[8]:


print(tools[0].description)


# Initialize an agent

# In[18]:


llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    handle_parsing_errors=True,
)


# In[12]:


# Extract pdf content
agent.run("What is the text on page 3 of the pdf?")


# In[13]:


# Simple Queries
agent.run("What was the US GDP in 2019?")


# In[14]:


# Calculations
agent.run("What would the GDP be in 2030 if the latest GDP number grew by 50%?")


# In[19]:


# Chart output
agent.run("Create a nice and labeled chart of the GDP growth over time")


# In[ ]:




