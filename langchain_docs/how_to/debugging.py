#!/usr/bin/env python
# coding: utf-8

# # How to debug your LLM apps
#
# Like building any type of software, at some point you'll need to debug when building with LLMs. A model call will fail, or model output will be misformatted, or there will be some nested model calls and it won't be clear where along the way an incorrect output was created.
#
# There are three main methods for debugging:
#
# - Verbose Mode: This adds print statements for "important" events in your chain.
# - Debug Mode: This add logging statements for ALL events in your chain.
# - LangSmith Tracing: This logs events to [LangSmith](https://docs.smith.langchain.com/) to allow for visualization there.
#
# |                        | Verbose Mode | Debug Mode | LangSmith Tracing |
# |------------------------|--------------|------------|-------------------|
# | Free                   | ✅            | ✅          | ✅                 |
# | UI                     | ❌            | ❌          | ✅                 |
# | Persisted              | ❌            | ❌          | ✅                 |
# | See all events         | ❌            | ✅          | ✅                 |
# | See "important" events | ✅            | ❌          | ✅                 |
# | Runs Locally           | ✅            | ✅          | ❌                 |
#
#
# ## Tracing
#
# Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls.
# As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent.
# The best way to do this is with [LangSmith](https://smith.langchain.com).
#
# After you sign up at the link above, make sure to set your environment variables to start logging traces:
#
# ```shell
# export LANGSMITH_TRACING="true"
# export LANGSMITH_API_KEY="..."
# ```
#
# Or, if in a notebook, you can set them with:
#
# ```python
# import getpass
# import os
#
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
# ```
#
# Let's suppose we have an agent, and want to visualize the actions it takes and tool outputs it receives. Without any debugging, here's what we see:
#
# import ChatModelTabs from "@theme/ChatModelTabs";
#
# <ChatModelTabs
#   customVarName="llm"
# />
#

# In[3]:


# | output: false
# | echo: false

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)


# In[4]:


from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

tools = [TavilySearchResults(max_results=1)]
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Construct the Tools agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools)
agent_executor.invoke(
    {"input": "Who directed the 2023 film Oppenheimer and what is their age in days?"}
)


# We don't get much output, but since we set up LangSmith we can easily see what happened under the hood:
#
# https://smith.langchain.com/public/a89ff88f-9ddc-4757-a395-3a1b365655bf/r

# ## `set_debug` and `set_verbose`
#
# If you're prototyping in Jupyter Notebooks or running Python scripts, it can be helpful to print out the intermediate steps of a chain run.
#
# There are a number of ways to enable printing at varying degrees of verbosity.
#
# Note: These still work even with LangSmith enabled, so you can have both turned on and running at the same time
#

# ### `set_verbose(True)`
#
# Setting the `verbose` flag will print out inputs and outputs in a slightly more readable format and will skip logging certain raw outputs (like the token usage stats for an LLM call) so that you can focus on application logic.

# In[6]:


from langchain.globals import set_verbose

set_verbose(True)
agent_executor = AgentExecutor(agent=agent, tools=tools)
agent_executor.invoke(
    {"input": "Who directed the 2023 film Oppenheimer and what is their age in days?"}
)


# ### `set_debug(True)`
#
# Setting the global `debug` flag will cause all LangChain components with callback support (chains, models, agents, tools, retrievers) to print the inputs they receive and outputs they generate. This is the most verbose setting and will fully log raw inputs and outputs.

# In[7]:


from langchain.globals import set_debug

set_debug(True)
set_verbose(False)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_executor.invoke(
    {"input": "Who directed the 2023 film Oppenheimer and what is their age in days?"}
)
