#!/usr/bin/env python
# coding: utf-8

# # Jira Toolkit
# 
# This notebook goes over how to use the `Jira` toolkit.
# 
# The `Jira` toolkit allows agents to interact with a given Jira instance, performing actions such as searching for issues and creating issues, the tool wraps the atlassian-python-api library, for more see: https://atlassian-python-api.readthedocs.io/jira.html
# 
# ## Installation and setup
# 
# To use this tool, you must first set as environment variables:
#     JIRA_API_TOKEN
#     JIRA_USERNAME
#     JIRA_INSTANCE_URL
#     JIRA_CLOUD

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  atlassian-python-api')


# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-community langchain_openai')


# In[2]:


import os

from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_openai import OpenAI


# In[3]:


os.environ["JIRA_API_TOKEN"] = "abc"
os.environ["JIRA_USERNAME"] = "123"
os.environ["JIRA_INSTANCE_URL"] = "https://jira.atlassian.com"
os.environ["OPENAI_API_KEY"] = "xyz"
os.environ["JIRA_CLOUD"] = "True"


# In[4]:


llm = OpenAI(temperature=0)
jira = JiraAPIWrapper()
toolkit = JiraToolkit.from_jira_api_wrapper(jira)


# ## Tool usage
# 
# Let's see what individual tools are in the Jira toolkit:

# In[5]:


[(tool.name, tool.description) for tool in toolkit.get_tools()]


# In[5]:


agent = initialize_agent(
    toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


# In[9]:


agent.run("make a new issue in project PW to remind me to make more fried rice")

