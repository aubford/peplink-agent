#!/usr/bin/env python
# coding: utf-8

# # How to force models to call a tool
# 
# :::info Prerequisites
# 
# This guide assumes familiarity with the following concepts:
# - [Chat models](/docs/concepts/chat_models)
# - [LangChain Tools](/docs/concepts/tools)
# - [How to use a model to call tools](/docs/how_to/tool_calling)
# :::
# 
# In order to force our LLM to select a specific [tool](/docs/concepts/tools/), we can use the `tool_choice` parameter to ensure certain behavior. First, let's define our model and tools:

# In[ ]:


from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]


# In[ ]:


# | output: false
# | echo: false

import os
from getpass import getpass

from langchain_openai import ChatOpenAI

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# For example, we can force our tool to call the multiply tool by using the following code:

# In[ ]:


llm_forced_to_multiply = llm.bind_tools(tools, tool_choice="multiply")
llm_forced_to_multiply.invoke("what is 2 + 4")


# Even if we pass it something that doesn't require multiplcation - it will still call the tool!

# We can also just force our tool to select at least one of our tools by passing in the "any" (or "required" which is OpenAI specific) keyword to the `tool_choice` parameter.

# In[ ]:


llm_forced_to_use_tool = llm.bind_tools(tools, tool_choice="any")
llm_forced_to_use_tool.invoke("What day is today?")

