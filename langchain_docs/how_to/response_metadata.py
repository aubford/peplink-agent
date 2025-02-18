#!/usr/bin/env python
# coding: utf-8

# # Response metadata
#
# Many model providers include some metadata in their chat generation [responses](/docs/concepts/messages/#aimessage). This metadata can be accessed via the `AIMessage.response_metadata: Dict` attribute. Depending on the model provider and model configuration, this can contain information like [token counts](/docs/how_to/chat_token_usage_tracking), [logprobs](/docs/how_to/logprobs), and more.
#
# Here's what the response metadata looks like for a few different providers:
#
# ## OpenAI

# In[2]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4-turbo")
msg = llm.invoke([("human", "What's the oldest known example of cuneiform")])
msg.response_metadata


# ## Anthropic

# In[7]:


from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-sonnet-20240229")
msg = llm.invoke([("human", "What's the oldest known example of cuneiform")])
msg.response_metadata


# ## Google VertexAI

# In[8]:


from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model="gemini-pro")
msg = llm.invoke([("human", "What's the oldest known example of cuneiform")])
msg.response_metadata


# ## Bedrock (Anthropic)

# In[16]:


from langchain_aws import ChatBedrock

llm = ChatBedrock(model_id="anthropic.claude-v2")
msg = llm.invoke([("human", "What's the oldest known example of cuneiform")])
msg.response_metadata


# ## MistralAI

# In[7]:


from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI()
msg = llm.invoke([("human", "What's the oldest known example of cuneiform")])
msg.response_metadata


# ## Groq

# In[1]:


from langchain_groq import ChatGroq

llm = ChatGroq()
msg = llm.invoke([("human", "What's the oldest known example of cuneiform")])
msg.response_metadata


# ## TogetherAI

# In[2]:


import os

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
)
msg = llm.invoke([("human", "What's the oldest known example of cuneiform")])
msg.response_metadata


# ## FireworksAI

# In[31]:


from langchain_fireworks import ChatFireworks

llm = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct")
msg = llm.invoke([("human", "What's the oldest known example of cuneiform")])
msg.response_metadata
