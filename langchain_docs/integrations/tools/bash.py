#!/usr/bin/env python
# coding: utf-8

# # Shell (bash)
#
# Giving agents access to the shell is powerful (though risky outside a sandboxed environment).
#
# The LLM can use it to execute any shell commands. A common use case for this is letting the LLM interact with your local file system.
#
# **Note:** Shell tool does not work with Windows OS.

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet langchain-community")


# In[ ]:


from langchain_community.tools import ShellTool

shell_tool = ShellTool()


# In[2]:


print(shell_tool.run({"commands": ["echo 'Hello World!'", "time"]}))


# ### Use with Agents
#
# As with all tools, these can be given to an agent to accomplish more complex tasks. Let's have the agent fetch some links from a web page.

# In[3]:


from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
    "{", "{{"
).replace("}", "}}")
self_ask_with_search = initialize_agent(
    [shell_tool], llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
self_ask_with_search.run(
    "Download the langchain.com webpage and grep for all urls. Return only a sorted list of them. Be sure to use double quotes."
)


# In[ ]:
