#!/usr/bin/env python
# coding: utf-8

# # How to parse text from message objects
#
# :::info Prerequisites
#
# This guide assumes familiarity with the following concepts:
# - [Chat models](/docs/concepts/chat_models/)
# - [Messages](/docs/concepts/messages/)
# - [Output parsers](/docs/concepts/output_parsers/)
# - [LangChain Expression Language (LCEL)](/docs/concepts/lcel/)
#
# :::
#
# LangChain [message](/docs/concepts/messages/) objects support content in a [variety of formats](/docs/concepts/messages/#content), including text, [multimodal data](/docs/concepts/multimodality/), and a list of [content block](/docs/concepts/messages/#aimessage) dicts.
#
# The format of [Chat model](/docs/concepts/chat_models/) response content may depend on the provider. For example, the chat model for [Anthropic](/docs/integrations/chat/anthropic/) will return string content for typical string input:

# In[1]:


from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-5-haiku-latest")

response = llm.invoke("Hello")
response.content


# But when tool calls are generated, the response content is structured into content blocks that convey the model's reasoning process:

# In[2]:


from langchain_core.tools import tool


@tool
def get_weather(location: str) -> str:
    """Get the weather from a location."""

    return "Sunny."


llm_with_tools = llm.bind_tools([get_weather])

response = llm_with_tools.invoke("What's the weather in San Francisco, CA?")
response.content


# To automatically parse text from message objects irrespective of the format of the underlying content, we can use [StrOutputParser](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html). We can compose it with a chat model as follows:

# In[3]:


from langchain_core.output_parsers import StrOutputParser

chain = llm_with_tools | StrOutputParser()


# [StrOutputParser](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html) simplifies the extraction of text from message objects:

# In[4]:


response = chain.invoke("What's the weather in San Francisco, CA?")
print(response)


# This is particularly useful in streaming contexts:

# In[5]:


for chunk in chain.stream("What's the weather in San Francisco, CA?"):
    print(chunk, end="|")


# See the [API Reference](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html) for more information.
