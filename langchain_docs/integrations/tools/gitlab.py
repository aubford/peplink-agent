#!/usr/bin/env python
# coding: utf-8

# # Gitlab Toolkit
# 
# The `Gitlab` toolkit contains tools that enable an LLM agent to interact with a gitlab repository. 
# The tool is a wrapper for the [python-gitlab](https://github.com/python-gitlab/python-gitlab) library. 
# 
# ## Quickstart
# 1. Install the python-gitlab library
# 2. Create a Gitlab personal access token
# 3. Set your environmental variables
# 4. Pass the tools to your agent with `toolkit.get_tools()`
# 
# Each of these steps will be explained in great detail below.
# 
# 1. **Get Issues**- fetches issues from the repository.
# 
# 2. **Get Issue**- fetches details about a specific issue.
# 
# 3. **Comment on Issue**- posts a comment on a specific issue.
# 
# 4. **Create Merge Request**- creates a merge request from the bot's working branch to the base branch.
# 
# 5. **Create File**- creates a new file in the repository.
# 
# 6. **Read File**- reads a file from the repository.
# 
# 7. **Update File**- updates a file in the repository.
# 
# 8. **Delete File**- deletes a file from the repository.
# 
# 

# ## Setup

# ### 1. Install the `python-gitlab` library 

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  python-gitlab langchain-community')


# ### 2. Create a Gitlab personal access token
# 
# [Follow the instructions here](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html) to create a Gitlab personal access token. Make sure your app has the following repository permissions:
# 
# * read_api
# * read_repository
# * write_repository
# 
# ### 3. Set Environmental Variables
# 
# Before initializing your agent, the following environmental variables need to be set:
# 
# * **GITLAB_URL** - The URL hosted Gitlab. Defaults to "https://gitlab.com". 
# * **GITLAB_PERSONAL_ACCESS_TOKEN**- The personal access token you created in the last step
# * **GITLAB_REPOSITORY**- The name of the Gitlab repository you want your bot to act upon. Must follow the format \{username\}/\{repo-name\}.
# * **GITLAB_BRANCH**- The branch where the bot will make its commits. Defaults to 'main.'
# * **GITLAB_BASE_BRANCH**- The base branch of your repo, usually either 'main' or 'master.' This is where merge requests will base from. Defaults to 'main.'
# 

# ## Example: Simple Agent

# In[3]:


import os

from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.gitlab.toolkit import GitLabToolkit
from langchain_community.utilities.gitlab import GitLabAPIWrapper
from langchain_openai import OpenAI


# In[53]:


# Set your environment variables using os.environ
os.environ["GITLAB_URL"] = "https://gitlab.example.org"
os.environ["GITLAB_PERSONAL_ACCESS_TOKEN"] = ""
os.environ["GITLAB_REPOSITORY"] = "username/repo-name"
os.environ["GITLAB_BRANCH"] = "bot-branch-name"
os.environ["GITLAB_BASE_BRANCH"] = "main"

# This example also requires an OpenAI API key
os.environ["OPENAI_API_KEY"] = ""


# In[5]:


llm = OpenAI(temperature=0)
gitlab = GitLabAPIWrapper()
toolkit = GitLabToolkit.from_gitlab_api_wrapper(gitlab)
agent = initialize_agent(
    toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


# In[8]:


agent.run(
    "You have the software engineering capabilities of a Google Principle engineer. You are tasked with completing issues on a gitlab repository. Please look at the open issues and complete them by creating merge requests that solve the issues."
)


# In[ ]:




