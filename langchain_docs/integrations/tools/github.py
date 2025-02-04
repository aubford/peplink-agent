#!/usr/bin/env python
# coding: utf-8

# # Github Toolkit
# 
# The `Github` toolkit contains tools that enable an LLM agent to interact with a github repository. 
# The tool is a wrapper for the [PyGitHub](https://github.com/PyGithub/PyGithub) library. 
# 
# For detailed documentation of all GithubToolkit features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.github.toolkit.GitHubToolkit.html).
# 
# ## Setup
# 
# At a high-level, we will:
# 
# 1. Install the pygithub library
# 2. Create a Github app
# 3. Set your environmental variables
# 4. Pass the tools to your agent with `toolkit.get_tools()`

# If you want to get automated tracing from runs of individual tools, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[ ]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# #### 1. Install dependencies
# 
# This integration is implemented in `langchain-community`. We will also need the `pygithub` dependency:

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  pygithub langchain-community')


# #### 2. Create a Github App
# 
# [Follow the instructions here](https://docs.github.com/en/apps/creating-github-apps/registering-a-github-app/registering-a-github-app) to create and register a Github app. Make sure your app has the following [repository permissions:](https://docs.github.com/en/rest/overview/permissions-required-for-github-apps?apiVersion=2022-11-28)
# 
# * Commit statuses (read only)
# * Contents (read and write)
# * Issues (read and write)
# * Metadata (read only)
# * Pull requests (read and write)
# 
# Once the app has been registered, you must give your app permission to access each of the repositories you whish it to act upon. Use the App settings on [github.com here](https://github.com/settings/installations).
# 
# 
# #### 3. Set Environment Variables
# 
# Before initializing your agent, the following environment variables need to be set:
# 
# * **GITHUB_APP_ID**- A six digit number found in your app's general settings
# * **GITHUB_APP_PRIVATE_KEY**- The location of your app's private key .pem file, or the full text of that file as a string.
# * **GITHUB_REPOSITORY**- The name of the Github repository you want your bot to act upon. Must follow the format \{username\}/\{repo-name\}. *Make sure the app has been added to this repository first!*
# * Optional: **GITHUB_BRANCH**- The branch where the bot will make its commits. Defaults to `repo.default_branch`.
# * Optional: **GITHUB_BASE_BRANCH**- The base branch of your repo upon which PRs will based from. Defaults to `repo.default_branch`.

# In[ ]:


import getpass
import os

for env_var in [
    "GITHUB_APP_ID",
    "GITHUB_APP_PRIVATE_KEY",
    "GITHUB_REPOSITORY",
]:
    if not os.getenv(env_var):
        os.environ[env_var] = getpass.getpass()


# ## Instantiation
# 
# Now we can instantiate our toolkit:

# In[2]:


from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper

github = GitHubAPIWrapper()
toolkit = GitHubToolkit.from_github_api_wrapper(github)


# ## Tools
# 
# View available tools:

# In[3]:


tools = toolkit.get_tools()

for tool in tools:
    print(tool.name)


# The purpose of these tools is as follows:
# 
# Each of these steps will be explained in great detail below.
# 
# 1. **Get Issues**- fetches issues from the repository.
# 
# 2. **Get Issue**- fetches details about a specific issue.
# 
# 3. **Comment on Issue**- posts a comment on a specific issue.
# 
# 4. **Create Pull Request**- creates a pull request from the bot's working branch to the base branch.
# 
# 5. **Create File**- creates a new file in the repository.
# 
# 6. **Read File**- reads a file from the repository.
# 
# 7. **Update File**- updates a file in the repository.
# 
# 8. **Delete File**- deletes a file from the repository.

# ## Include release tools
# 
# By default, the toolkit does not include release-related tools. You can include them by setting `include_release_tools=True` when initializing the toolkit:

# In[ ]:


toolkit = GitHubToolkit.from_github_api_wrapper(github, include_release_tools=True)


# Settings `include_release_tools=True` will include the following tools:
# 
# * **Get Latest Release**- fetches the latest release from the repository.
# 
# * **Get Releases**- fetches the latest 5 releases from the repository.
# 
# * **Get Release**- fetches a specific release from the repository by tag name, e.g. `v1.0.0`.
# 

# ## Use within an agent
# 
# We will need a LLM or chat model:
# 
# import ChatModelTabs from "@theme/ChatModelTabs";
# 
# <ChatModelTabs customVarName="llm" />
# 

# In[4]:


# | output: false
# | echo: false

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Initialize the agent with a subset of tools:

# In[6]:


from langgraph.prebuilt import create_react_agent

tools = [tool for tool in toolkit.get_tools() if tool.name == "Get Issue"]
assert len(tools) == 1
tools[0].name = "get_issue"

agent_executor = create_react_agent(llm, tools)


# And issue it a query:

# In[7]:


example_query = "What is the title of issue 24888?"

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()


# ## API reference
# 
# For detailed documentation of all `GithubToolkit` features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.github.toolkit.GitHubToolkit.html).
