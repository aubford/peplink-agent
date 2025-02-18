#!/usr/bin/env python
# coding: utf-8

# # ClickUp Toolkit
#
# >[ClickUp](https://clickup.com/) is an all-in-one productivity platform that provides small and large teams across industries with flexible and customizable work management solutions, tools, and functions.
#
# >It is a cloud-based project management solution for businesses of all sizes featuring communication and collaboration tools to help achieve organizational goals.

# In[ ]:


get_ipython().run_line_magic("pip", "install -qU langchain-community")


# In[1]:


get_ipython().run_line_magic("reload_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
from datetime import datetime

from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.clickup.toolkit import ClickupToolkit
from langchain_community.utilities.clickup import ClickupAPIWrapper
from langchain_openai import OpenAI


# ## Initializing

# ### Get Authenticated
# 1. Create a [ClickUp App](https://help.clickup.com/hc/en-us/articles/6303422883095-Create-your-own-app-with-the-ClickUp-API)
# 2. Follow [these steps](https://clickup.com/api/developer-portal/authentication/) to get your `client_id` and `client_secret`.
#     - *Suggestion: use `https://google.com` as the redirect_uri. This is what we assume in the defaults for this toolkit.*
# 3. Copy/paste them and run the next cell to get your `code`
#

# In[48]:


# Copilot Sandbox
oauth_client_id = "ABC..."
oauth_client_secret = "123..."
redirect_uri = "https://google.com"

print("Click this link, select your workspace, click `Connect Workspace`")
print(ClickupAPIWrapper.get_access_code_url(oauth_client_id, redirect_uri))


# The url should change to something like this https://www.google.com/?code=THISISMYCODERIGHTHERE.
#
# Next, copy/paste the `CODE` (THISISMYCODERIGHTHERE) generated in the URL in the cell below.
#

# In[4]:


code = "THISISMYCODERIGHTHERE"


# ### Get Access Token
# Then, use the code below to get your `access_token`.
#
# *Important*: Each code is a one time code that will expire after use. The `access_token` can be used for a period of time. Make sure to copy paste the `access_token` once you get it!

# In[5]:


access_token = ClickupAPIWrapper.get_access_token(
    oauth_client_id, oauth_client_secret, code
)


# In[3]:


# Init toolkit
clickup_api_wrapper = ClickupAPIWrapper(access_token=access_token)
toolkit = ClickupToolkit.from_clickup_api_wrapper(clickup_api_wrapper)
print(
    f"Found team_id: {clickup_api_wrapper.team_id}.\nMost request require the team id, so we store it for you in the toolkit, we assume the first team in your list is the one you want. \nNote: If you know this is the wrong ID, you can pass it at initialization."
)


# ### Create Agent

# In[5]:


llm = OpenAI(temperature=0, openai_api_key="")

agent = initialize_agent(
    toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


# ## Use an Agent

# In[6]:


# helper function for demo
def print_and_run(command):
    print("\033[94m$ COMMAND\033[0m")
    print(command)
    print("\n\033[94m$ AGENT\033[0m")
    response = agent.run(command)
    print("".join(["-"] * 80))
    return response


# ### Navigation
# You can get the teams, folder and spaces your user has access to

# In[7]:


print_and_run("Get all the teams that the user is authorized to access")
print_and_run("Get all the spaces available to the team")
print_and_run("Get all the folders for the team")


# ### Task Operations
# You can get, ask question about tasks and update them

# In[32]:


task_id = "8685mb5fn"


# #### Basic attirbute getting and updating

# In[33]:


# We can get a task to inspect it's contents
print_and_run(f"Get task with id {task_id}")

# We can get a specific attribute from a task
previous_description = print_and_run(
    f"What is the description of task with id {task_id}"
)

# We can even update it!
print_and_run(
    f"For task with id {task_id}, change the description to 'A cool task descriptiont changed by AI!'"
)
print_and_run(f"What is the description of task with id {task_id}")

# Undo what we did
print_and_run(
    f"For task with id {task_id}, change the description to '{previous_description}'"
)


# In[35]:


print_and_run("Change the descrition task 8685mj6cd to 'Look ma no hands'")


# #### Advanced Attributes (Assignees)
# You can query and update almost every thing about a task!

# In[36]:


user_id = 81928627


# In[37]:


print_and_run(f"What are the assignees of task id {task_id}?")
print_and_run(f"Remove user {user_id} from the assignees of task id {task_id}")
print_and_run(f"What are the assignees of task id {task_id}?")
print_and_run(f"Add user {user_id} from the assignees of task id {task_id}")


# ### Creation
# You can create tasks, lists and folders

# In[27]:


time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
print_and_run(
    f"Create a task called 'Test Task - {time_str}' with description 'This is a Test'"
)


# In[29]:


time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
print_and_run(f"Create a list called Test List - {time_str}")


# In[30]:


time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
print_and_run(f"Create a folder called 'Test Folder - {time_str}'")


# In[32]:


time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
print_and_run(
    f"Create a list called 'Test List - {time_str}' with content My test list with high priority and status red"
)


# ## Multi-Step Tasks

# In[34]:


print_and_run(
    "Figure out what user ID Rodrigo is, create a task called 'Rod's task', assign it to Rodrigo"
)
