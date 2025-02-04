#!/usr/bin/env python
# coding: utf-8

# # How to add tools to chatbots
# 
# :::info Prerequisites
# 
# This guide assumes familiarity with the following concepts:
# 
# - [Chatbots](/docs/concepts/messages)
# - [Agents](/docs/tutorials/agents)
# - [Chat history](/docs/concepts/chat_history)
# 
# :::
# 
# This section will cover how to create conversational agents: chatbots that can interact with other systems and APIs using tools.
# 
# :::note
# 
# This how-to guide previously built a chatbot using [RunnableWithMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html). You can access this version of the guide in the [v0.2 docs](https://python.langchain.com/v0.2/docs/how_to/chatbots_tools/).
# 
# As of the v0.3 release of LangChain, we recommend that LangChain users take advantage of [LangGraph persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) to incorporate `memory` into new LangChain applications.
# 
# If your code is already relying on `RunnableWithMessageHistory` or `BaseChatMessageHistory`, you do **not** need to make any changes. We do not plan on deprecating this functionality in the near future as it works for simple chat applications and any code that uses `RunnableWithMessageHistory` will continue to work as expected.
# 
# Please see [How to migrate to LangGraph Memory](/docs/versions/migrating_memory/) for more details.
# :::
# 
# ## Setup
# 
# For this guide, we'll be using a [tool calling agent](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#tool-calling-agent) with a single tool for searching the web. The default will be powered by [Tavily](/docs/integrations/tools/tavily_search), but you can switch it out for any similar tool. The rest of this section will assume you're using Tavily.
# 
# You'll need to [sign up for an account](https://tavily.com/) on the Tavily website, and install the following packages:

# In[1]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet langchain-community langchain-openai tavily-python langgraph')

import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API Key:")


# You will also need your OpenAI key set as `OPENAI_API_KEY` and your Tavily API key set as `TAVILY_API_KEY`.

# ## Creating an agent
# 
# Our end goal is to create an agent that can respond conversationally to user questions while looking up information as needed.
# 
# First, let's initialize Tavily and an OpenAI [chat model](/docs/concepts/chat_models/) capable of tool calling:

# In[2]:


from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

tools = [TavilySearchResults(max_results=1)]

# Choose the LLM that will drive the agent
# Only certain models support this
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# To make our agent conversational, we can also specify a prompt. Here's an example:

# In[3]:


prompt = (
    "You are a helpful assistant. "
    "You may not need to use tools for every query - the user may just want to chat!"
)


# Great! Now let's assemble our agent using LangGraph's prebuilt [create_react_agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent), which allows you to create a [tool-calling agent](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#tool-calling-agent):

# In[4]:


from langgraph.prebuilt import create_react_agent

# prompt allows you to preprocess the inputs to the model inside ReAct agent
# in this case, since we're passing a prompt string, we'll just always add a SystemMessage
# with this prompt string before any other messages sent to the model
agent = create_react_agent(model, tools, prompt=prompt)


# ## Running the agent
# 
# Now that we've set up our agent, let's try interacting with it! It can handle both trivial queries that require no lookup:

# In[5]:


from langchain_core.messages import HumanMessage

agent.invoke({"messages": [HumanMessage(content="I'm Nemo!")]})


# Or, it can use of the passed search tool to get up to date information if needed:

# In[6]:


agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="What is the current conservation status of the Great Barrier Reef?"
            )
        ],
    }
)


# ## Conversational responses
# 
# Because our prompt contains a placeholder for chat history messages, our agent can also take previous interactions into account and respond conversationally like a standard chatbot:

# In[7]:


from langchain_core.messages import AIMessage, HumanMessage

agent.invoke(
    {
        "messages": [
            HumanMessage(content="I'm Nemo!"),
            AIMessage(content="Hello Nemo! How can I assist you today?"),
            HumanMessage(content="What is my name?"),
        ],
    }
)


# If preferred, you can also add memory to the LangGraph agent to manage the history of messages. Let's redeclare it this way:

# In[8]:


from langgraph.checkpoint.memory import MemorySaver

# highlight-start
memory = MemorySaver()
agent = create_react_agent(model, tools, prompt=prompt, checkpointer=memory)
# highlight-end


# In[9]:


agent.invoke(
    {"messages": [HumanMessage("I'm Nemo!")]},
    config={"configurable": {"thread_id": "1"}},
)


# And then if we rerun our wrapped agent executor:

# In[10]:


agent.invoke(
    {"messages": [HumanMessage("What is my name?")]},
    config={"configurable": {"thread_id": "1"}},
)


# This [LangSmith trace](https://smith.langchain.com/public/9e6b000d-08aa-4c5a-ac83-2fdf549523cb/r) shows what's going on under the hood.
# 
# ## Further reading
# 
# For more on how to build agents, check these [LangGraph](https://langchain-ai.github.io/langgraph/) guides:
# 
# * [agents conceptual guide](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/)
# * [agents tutorials](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/)
# * [create_react_agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/)
# 
# For more on tool usage, you can also check out [this use case section](/docs/how_to#tools).
