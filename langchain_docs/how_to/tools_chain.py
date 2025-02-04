#!/usr/bin/env python
# coding: utf-8
---
sidebar_position: 0
---
# # How to use tools in a chain
# 
# In this guide, we will go over the basic ways to create Chains and Agents that call [Tools](/docs/concepts/tools/). Tools can be just about anything — APIs, functions, databases, etc. Tools allow us to extend the capabilities of a model beyond just outputting text/messages. The key to using models with tools is correctly prompting a model and parsing its response so that it chooses the right tools and provides the right inputs for them.

# ## Setup
# 
# We'll need to install the following packages for this guide:

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet langchain')


# If you'd like to trace your runs in [LangSmith](https://docs.smith.langchain.com/) uncomment and set the following environment variables:

# In[ ]:


import getpass
import os

# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()


# ## Create a tool
# 
# First, we need to create a tool to call. For this example, we will create a custom tool from a function. For more information on creating custom tools, please see [this guide](/docs/how_to/custom_tools).

# In[6]:


from langchain_core.tools import tool


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


# In[2]:


print(multiply.name)
print(multiply.description)
print(multiply.args)


# In[3]:


multiply.invoke({"first_int": 4, "second_int": 5})


# ## Chains
# 
# If we know that we only need to use a tool a fixed number of times, we can create a chain for doing so. Let's create a simple chain that just multiplies user-specified numbers.
# 
# ![chain](../../static/img/tool_chain.svg)
# 
# ### Tool/function calling
# One of the most reliable ways to use tools with LLMs is with [tool calling](/docs/concepts/tool_calling/) APIs (also sometimes called function calling). This only works with models that explicitly support tool calling. You can see which models support tool calling [here](/docs/integrations/chat/), and learn more about how to use tool calling in [this guide](/docs/how_to/function_calling).
# 
# First we'll define our model and tools. We'll start with just a single tool, `multiply`.
# 
# import ChatModelTabs from "@theme/ChatModelTabs";
# 
# <ChatModelTabs customVarName="llm"/>
# 

# In[7]:


# | echo: false
# | output: false

from langchain_openai.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# We'll use `bind_tools` to pass the definition of our tool in as part of each call to the model, so that the model can invoke the tool when appropriate:

# In[8]:


llm_with_tools = llm.bind_tools([multiply])


# When the model invokes the tool, this will show up in the `AIMessage.tool_calls` attribute of the output:

# In[9]:


msg = llm_with_tools.invoke("whats 5 times forty two")
msg.tool_calls


# Check out the [LangSmith trace here](https://smith.langchain.com/public/81ff0cbd-e05b-4720-bf61-2c9807edb708/r).

# ### Invoking the tool
# 
# Great! We're able to generate tool invocations. But what if we want to actually call the tool? To do so we'll need to pass the generated tool args to our tool. As a simple example we'll just extract the arguments of the first tool_call:

# In[12]:


from operator import itemgetter

chain = llm_with_tools | (lambda x: x.tool_calls[0]["args"]) | multiply
chain.invoke("What's four times 23")


# Check out the [LangSmith trace here](https://smith.langchain.com/public/16bbabb9-fc9b-41e5-a33d-487c42df4f85/r).

# ## Agents
# 
# Chains are great when we know the specific sequence of tool usage needed for any user input. But for certain use cases, how many times we use tools depends on the input. In these cases, we want to let the model itself decide how many times to use tools and in what order. [Agents](/docs/tutorials/agents) let us do just this.
# 
# LangChain comes with a number of built-in agents that are optimized for different use cases. Read about all the [agent types here](/docs/concepts/agents).
# 
# We'll use the [tool calling agent](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.tool_calling_agent.base.create_tool_calling_agent.html), which is generally the most reliable kind and the recommended one for most use cases.
# 
# ![agent](../../static/img/tool_agent.svg)

# In[13]:


from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent


# In[14]:


# Get the prompt to use - can be replaced with any prompt that includes variables "agent_scratchpad" and "input"!
prompt = hub.pull("hwchase17/openai-tools-agent")
prompt.pretty_print()


# Agents are also great because they make it easy to use multiple tools.

# In[15]:


@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent


tools = [multiply, add, exponentiate]


# In[16]:


# Construct the tool calling agent
agent = create_tool_calling_agent(llm, tools, prompt)


# In[17]:


# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# With an agent, we can ask questions that require arbitrarily-many uses of our tools:

# In[18]:


agent_executor.invoke(
    {
        "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"
    }
)


# Check out the [LangSmith trace here](https://smith.langchain.com/public/eeeb27a4-a2f8-4f06-a3af-9c983f76146c/r).
