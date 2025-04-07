#!/usr/bin/env python
# coding: utf-8

# # How to filter messages
# 
# In more complex chains and agents we might track state with a list of [messages](/docs/concepts/messages/). This list can start to accumulate messages from multiple different models, speakers, sub-chains, etc., and we may only want to pass subsets of this full list of messages to each model call in the chain/agent.
# 
# The `filter_messages` utility makes it easy to filter messages by type, id, or name.
# 
# ## Basic usage

# In[1]:


from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    filter_messages,
)

messages = [
    SystemMessage("you are a good assistant", id="1"),
    HumanMessage("example input", id="2", name="example_user"),
    AIMessage("example output", id="3", name="example_assistant"),
    HumanMessage("real input", id="4", name="bob"),
    AIMessage("real output", id="5", name="alice"),
]

filter_messages(messages, include_types="human")


# In[2]:


filter_messages(messages, exclude_names=["example_user", "example_assistant"])


# In[3]:


filter_messages(messages, include_types=[HumanMessage, AIMessage], exclude_ids=["3"])


# ## Chaining
# 
# `filter_messages` can be used in an imperatively (like above) or declaratively, making it easy to compose with other components in a chain:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-anthropic')


# In[4]:


from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
# Notice we don't pass in messages. This creates
# a RunnableLambda that takes messages as input
filter_ = filter_messages(exclude_names=["example_user", "example_assistant"])
chain = filter_ | llm
chain.invoke(messages)


# Looking at the LangSmith trace we can see that before the messages are passed to the model they are filtered: https://smith.langchain.com/public/f808a724-e072-438e-9991-657cc9e7e253/r
# 
# Looking at just the filter_, we can see that it's a Runnable object that can be invoked like all Runnables:

# In[6]:


filter_.invoke(messages)


# ## API reference
# 
# For a complete description of all arguments head to the API reference: https://python.langchain.com/api_reference/core/messages/langchain_core.messages.utils.filter_messages.html
