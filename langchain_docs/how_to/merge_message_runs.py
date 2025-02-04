#!/usr/bin/env python
# coding: utf-8

# # How to merge consecutive messages of the same type
# 
# Certain models do not support passing in consecutive [messages](/docs/concepts/messages/) of the same type (a.k.a. "runs" of the same message type).
# 
# The `merge_message_runs` utility makes it easy to merge consecutive messages of the same type.
# 
# ### Setup

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-core langchain-anthropic')


# ## Basic usage

# In[8]:


from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)

messages = [
    SystemMessage("you're a good assistant."),
    SystemMessage("you always respond with a joke."),
    HumanMessage([{"type": "text", "text": "i wonder why it's called langchain"}]),
    HumanMessage("and who is harrison chasing anyways"),
    AIMessage(
        'Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!'
    ),
    AIMessage("Why, he's probably chasing after the last cup of coffee in the office!"),
]

merged = merge_message_runs(messages)
print("\n\n".join([repr(x) for x in merged]))


# Notice that if the contents of one of the messages to merge is a list of content blocks then the merged message will have a list of content blocks. And if both messages to merge have string contents then those are concatenated with a newline character.

# ## Chaining
# 
# `merge_message_runs` can be used in an imperatively (like above) or declaratively, making it easy to compose with other components in a chain:

# In[9]:


from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
# Notice we don't pass in messages. This creates
# a RunnableLambda that takes messages as input
merger = merge_message_runs()
chain = merger | llm
chain.invoke(messages)


# Looking at the LangSmith trace we can see that before the messages are passed to the model they are merged: https://smith.langchain.com/public/ab558677-cac9-4c59-9066-1ecce5bcd87c/r
# 
# Looking at just the merger, we can see that it's a Runnable object that can be invoked like all Runnables:

# In[10]:


merger.invoke(messages)


# `merge_message_runs` can also be placed after a prompt:

# In[14]:


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate(
    [
        ("system", "You're great a {skill}"),
        ("system", "You're also great at explaining things"),
        ("human", "{query}"),
    ]
)
chain = prompt | merger | llm
chain.invoke({"skill": "math", "query": "what's the definition of a convergent series"})


# LangSmith Trace: https://smith.langchain.com/public/432150b6-9909-40a7-8ae7-944b7e657438/r/f4ad5fb2-4d38-42a6-b780-25f62617d53f

# ## API reference
# 
# For a complete description of all arguments head to the API reference: https://python.langchain.com/api_reference/core/messages/langchain_core.messages.utils.merge_message_runs.html
