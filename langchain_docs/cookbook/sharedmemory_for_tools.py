#!/usr/bin/env python
# coding: utf-8

# # Shared memory across agents and tools
# 
# This notebook goes over adding memory to **both** an Agent and its tools. Before going through this notebook, please walk through the following notebooks, as this will build on top of both of them:
# 
# - [Adding memory to an LLM Chain](/docs/modules/memory/integrations/adding_memory)
# - [Custom Agents](/docs/modules/agents/how_to/custom_agent)
# 
# We are going to create a custom Agent. The agent has access to a conversation memory, search tool, and a summarization tool. The summarization tool also needs access to the conversation memory.

# In[1]:


from langchain import hub
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent, create_react_agent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.prompts import PromptTemplate
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_openai import OpenAI


# In[2]:


template = """This is a conversation between a human and a bot:

{chat_history}

Write a summary of the conversation for {input}:
"""

prompt = PromptTemplate(input_variables=["input", "chat_history"], template=template)
memory = ConversationBufferMemory(memory_key="chat_history")
readonlymemory = ReadOnlySharedMemory(memory=memory)
summary_chain = LLMChain(
    llm=OpenAI(),
    prompt=prompt,
    verbose=True,
    memory=readonlymemory,  # use the read-only memory to prevent the tool from modifying the memory
)


# In[3]:


search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="Summary",
        func=summary_chain.run,
        description="useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary.",
    ),
]


# In[4]:


prompt = hub.pull("hwchase17/react")


# We can now construct the `LLMChain`, with the Memory object, and then create the agent.

# In[5]:


model = OpenAI()
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)


# In[36]:


agent_executor.invoke({"input": "What is ChatGPT?"})


# To test the memory of this agent, we can ask a followup question that relies on information in the previous exchange to be answered correctly.

# In[7]:


agent_executor.invoke({"input": "Who developed it?"})


# In[8]:


agent_executor.invoke(
    {"input": "Thanks. Summarize the conversation, for my daughter 5 years old."}
)


# Confirm that the memory was correctly updated.

# In[9]:


print(agent_executor.memory.buffer)


# 

# For comparison, below is a bad example that uses the same memory for both the Agent and the tool.

# In[10]:


## This is a bad practice for using the memory.
## Use the ReadOnlySharedMemory class, as shown above.

template = """This is a conversation between a human and a bot:

{chat_history}

Write a summary of the conversation for {input}:
"""

prompt = PromptTemplate(input_variables=["input", "chat_history"], template=template)
memory = ConversationBufferMemory(memory_key="chat_history")
summary_chain = LLMChain(
    llm=OpenAI(),
    prompt=prompt,
    verbose=True,
    memory=memory,  # <--- this is the only change
)

search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="Summary",
        func=summary_chain.run,
        description="useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary.",
    ),
]

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)


# In[11]:


agent_executor.invoke({"input": "What is ChatGPT?"})


# In[12]:


agent_executor.invoke({"input": "Who developed it?"})


# In[13]:


agent_executor.invoke(
    {"input": "Thanks. Summarize the conversation, for my daughter 5 years old."}
)


# The final answer is not wrong, but we see the 3rd Human input is actually from the agent in the memory because the memory was modified by the summary tool.

# In[14]:


print(agent_executor.memory.buffer)

