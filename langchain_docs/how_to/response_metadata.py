#!/usr/bin/env python
# coding: utf-8

# # Response metadata
# 
# Many model providers include some metadata in their chat generation [responses](/docs/concepts/messages/#aimessage). This metadata can be accessed via the `AIMessage.response_metadata: Dict` attribute. Depending on the model provider and model configuration, this can contain information like [token counts](/docs/how_to/chat_token_usage_tracking), [logprobs](/docs/how_to/logprobs), and more.
# 
# Here's what the response metadata looks like for a few different providers:
# 
# ## OpenAI

# In[1]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
msg = llm.invoke("What's the oldest known example of cuneiform")
msg.response_metadata


# ## Anthropic

# In[2]:


from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
msg = llm.invoke("What's the oldest known example of cuneiform")
msg.response_metadata


# ## Google VertexAI

# In[3]:


from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model="gemini-1.5-flash-001")
msg = llm.invoke("What's the oldest known example of cuneiform")
msg.response_metadata


# ## Bedrock (Anthropic)

# In[4]:


from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(model="anthropic.claude-3-sonnet-20240229-v1:0")
msg = llm.invoke("What's the oldest known example of cuneiform")
msg.response_metadata


# ## MistralAI

# In[5]:


from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(model="mistral-small-latest")
msg = llm.invoke([("human", "What's the oldest known example of cuneiform")])
msg.response_metadata


# ## Groq

# In[6]:


from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")
msg = llm.invoke("What's the oldest known example of cuneiform")
msg.response_metadata


# ## FireworksAI

# In[7]:


from langchain_fireworks import ChatFireworks

llm = ChatFireworks(model="accounts/fireworks/models/llama-v3p1-70b-instruct")
msg = llm.invoke("What's the oldest known example of cuneiform")
msg.response_metadata

