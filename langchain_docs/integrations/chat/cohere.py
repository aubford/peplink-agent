#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Cohere
---
# # Cohere
# 
# This notebook covers how to get started with [Cohere chat models](https://cohere.com/chat).
# 
# Head to the [API reference](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.cohere.ChatCohere.html) for detailed documentation of all attributes and methods.

# ## Setup
# 
# The integration lives in the `langchain-cohere` package. We can install these with:
# 
# ```bash
# pip install -U langchain-cohere
# ```
# 
# We'll also need to get a [Cohere API key](https://cohere.com/) and set the `COHERE_API_KEY` environment variable:

# In[11]:


import getpass
import os

os.environ["COHERE_API_KEY"] = getpass.getpass()


# It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability

# In[ ]:


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")


# ## Usage
# 
# ChatCohere supports all [ChatModel](/docs/how_to#chat-models) functionality:

# In[ ]:


from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage


# In[13]:


chat = ChatCohere()


# In[15]:


messages = [HumanMessage(content="1"), HumanMessage(content="2 3")]
chat.invoke(messages)


# In[16]:


await chat.ainvoke(messages)


# In[17]:


for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)


# In[18]:


chat.batch([messages])


# ## Chaining
# 
# You can also easily combine with a prompt template for easy structuring of user input. We can do this using [LCEL](/docs/concepts/lcel)

# In[19]:


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | chat


# In[20]:


chain.invoke({"topic": "bears"})


# ## Tool calling
# 
# Cohere supports tool calling functionalities!

# In[7]:


from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool


# In[8]:


@tool
def magic_function(number: int) -> int:
    """Applies a magic operation to an integer
    Args:
        number: Number to have magic operation performed on
    """
    return number + 10


def invoke_tools(tool_calls, messages):
    for tool_call in tool_calls:
        selected_tool = {"magic_function": magic_function}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
    return messages


tools = [magic_function]


# In[9]:


llm_with_tools = chat.bind_tools(tools=tools)
messages = [HumanMessage(content="What is the value of magic_function(2)?")]


# In[11]:


res = llm_with_tools.invoke(messages)
while res.tool_calls:
    messages.append(res)
    messages = invoke_tools(res.tool_calls, messages)
    res = llm_with_tools.invoke(messages)

res

