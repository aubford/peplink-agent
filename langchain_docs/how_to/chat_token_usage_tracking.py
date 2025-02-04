#!/usr/bin/env python
# coding: utf-8

# # How to track token usage in ChatModels
# 
# :::info Prerequisites
# 
# This guide assumes familiarity with the following concepts:
# - [Chat models](/docs/concepts/chat_models)
# 
# :::
# 
# Tracking [token](/docs/concepts/tokens/) usage to calculate cost is an important part of putting your app in production. This guide goes over how to obtain this information from your LangChain model calls.
# 
# This guide requires `langchain-anthropic` and `langchain-openai >= 0.1.9`.

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-anthropic langchain-openai')


# ## Using LangSmith
# 
# You can use [LangSmith](https://www.langchain.com/langsmith) to help track token usage in your LLM application. See the [LangSmith quick start guide](https://docs.smith.langchain.com/).
# 
# ## Using AIMessage.usage_metadata
# 
# A number of model providers return token usage information as part of the chat generation response. When available, this information will be included on the `AIMessage` objects produced by the corresponding model.
# 
# LangChain `AIMessage` objects include a [usage_metadata](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessage.html#langchain_core.messages.ai.AIMessage.usage_metadata) attribute. When populated, this attribute will be a [UsageMetadata](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.UsageMetadata.html) dictionary with standard keys (e.g., `"input_tokens"` and `"output_tokens"`).
# 
# Examples:
# 
# **OpenAI**:

# In[1]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
openai_response = llm.invoke("hello")
openai_response.usage_metadata


# **Anthropic**:

# In[2]:


from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-haiku-20240307")
anthropic_response = llm.invoke("hello")
anthropic_response.usage_metadata


# ### Using AIMessage.response_metadata
# 
# Metadata from the model response is also included in the AIMessage [response_metadata](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessage.html#langchain_core.messages.ai.AIMessage.response_metadata) attribute. These data are typically not standardized. Note that different providers adopt different conventions for representing token counts:

# In[3]:


print(f'OpenAI: {openai_response.response_metadata["token_usage"]}\n')
print(f'Anthropic: {anthropic_response.response_metadata["usage"]}')


# ### Streaming
# 
# Some providers support token count metadata in a streaming context.
# 
# #### OpenAI
# 
# For example, OpenAI will return a message [chunk](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessageChunk.html) at the end of a stream with token usage information. This behavior is supported by `langchain-openai >= 0.1.9` and can be enabled by setting `stream_usage=True`. This attribute can also be set when `ChatOpenAI` is instantiated.
# 
# :::note
# By default, the last message chunk in a stream will include a `"finish_reason"` in the message's `response_metadata` attribute. If we include token usage in streaming mode, an additional chunk containing usage metadata will be added to the end of the stream, such that `"finish_reason"` appears on the second to last message chunk.
# :::
# 

# In[4]:


llm = ChatOpenAI(model="gpt-4o-mini")

aggregate = None
for chunk in llm.stream("hello", stream_usage=True):
    print(chunk)
    aggregate = chunk if aggregate is None else aggregate + chunk


# Note that the usage metadata will be included in the sum of the individual message chunks:

# In[5]:


print(aggregate.content)
print(aggregate.usage_metadata)


# To disable streaming token counts for OpenAI, set `stream_usage` to False, or omit it from the parameters:

# In[6]:


aggregate = None
for chunk in llm.stream("hello"):
    print(chunk)


# You can also enable streaming token usage by setting `stream_usage` when instantiating the chat model. This can be useful when incorporating chat models into LangChain [chains](/docs/concepts/lcel): usage metadata can be monitored when [streaming intermediate steps](/docs/how_to/streaming#using-stream-events) or using tracing software such as [LangSmith](https://docs.smith.langchain.com/).
# 
# See the below example, where we return output structured to a desired schema, but can still observe token usage streamed from intermediate steps.

# In[8]:


from pydantic import BaseModel, Field


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


llm = ChatOpenAI(
    model="gpt-4o-mini",
    stream_usage=True,
)
# Under the hood, .with_structured_output binds tools to the
# chat model and appends a parser.
structured_llm = llm.with_structured_output(Joke)

async for event in structured_llm.astream_events("Tell me a joke", version="v2"):
    if event["event"] == "on_chat_model_end":
        print(f'Token usage: {event["data"]["output"].usage_metadata}\n')
    elif event["event"] == "on_chain_end":
        print(event["data"]["output"])
    else:
        pass


# Token usage is also visible in the corresponding [LangSmith trace](https://smith.langchain.com/public/fe6513d5-7212-4045-82e0-fefa28bc7656/r) in the payload from the chat model.

# ## Using callbacks
# 
# There are also some API-specific callback context managers that allow you to track token usage across multiple calls. They are currently only implemented for the OpenAI API and Bedrock Anthropic API, and are available in `langchain-community`:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-community')


# ### OpenAI
# 
# Let's first look at an extremely simple example of tracking token usage for a single Chat model call.

# In[9]:


from langchain_community.callbacks.manager import get_openai_callback

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    stream_usage=True,
)

with get_openai_callback() as cb:
    result = llm.invoke("Tell me a joke")
    print(cb)


# Anything inside the context manager will get tracked. Here's an example of using it to track multiple calls in sequence.

# In[10]:


with get_openai_callback() as cb:
    result = llm.invoke("Tell me a joke")
    result2 = llm.invoke("Tell me a joke")
    print(cb.total_tokens)


# In[11]:


with get_openai_callback() as cb:
    for chunk in llm.stream("Tell me a joke"):
        pass
    print(cb)


# If a chain or agent with multiple steps in it is used, it will track all those steps.

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain langchain-aws wikipedia')


# In[12]:


from langchain.agents import AgentExecutor, create_tool_calling_agent, load_tools
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're a helpful assistant"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
tools = load_tools(["wikipedia"])
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# In[13]:


with get_openai_callback() as cb:
    response = agent_executor.invoke(
        {
            "input": "What's a hummingbird's scientific name and what's the fastest bird species?"
        }
    )
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")


# ### Bedrock Anthropic
# 
# The `get_bedrock_anthropic_callback` works very similarly:

# In[12]:


from langchain_aws import ChatBedrock
from langchain_community.callbacks.manager import get_bedrock_anthropic_callback

llm = ChatBedrock(model_id="anthropic.claude-v2")

with get_bedrock_anthropic_callback() as cb:
    result = llm.invoke("Tell me a joke")
    result2 = llm.invoke("Tell me a joke")
    print(cb)


# ## Next steps
# 
# You've now seen a few examples of how to track token usage for supported providers.
# 
# Next, check out the other how-to guides chat models in this section, like [how to get a model to return structured output](/docs/how_to/structured_output) or [how to add caching to your chat models](/docs/how_to/chat_model_caching).

# In[ ]:




