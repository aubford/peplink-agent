#!/usr/bin/env python
# coding: utf-8

# # OpenAI Adapter
#
# **Please ensure OpenAI library is version 1.0.0 or higher; otherwise, refer to the older doc [OpenAI Adapter(Old)](/docs/integrations/adapters/openai-old/).**
#
# A lot of people get started with OpenAI but want to explore other models. LangChain's integrations with many model providers make this easy to do so. While LangChain has it's own message and model APIs, we've also made it as easy as possible to explore other models by exposing an adapter to adapt LangChain models to the OpenAI api.
#
# At the moment this only deals with output and does not return other information (token counts, stop reasons, etc).

# In[1]:


import openai
from langchain_community.adapters import openai as lc_openai


# ## chat.completions.create

# In[2]:


messages = [{"role": "user", "content": "hi"}]


# Original OpenAI call

# In[3]:


result = openai.chat.completions.create(
    messages=messages, model="gpt-3.5-turbo", temperature=0
)
result.choices[0].message.model_dump()


# LangChain OpenAI wrapper call

# In[4]:


lc_result = lc_openai.chat.completions.create(
    messages=messages, model="gpt-3.5-turbo", temperature=0
)

lc_result.choices[0].message  # Attribute access


# In[5]:


lc_result["choices"][0]["message"]  # Also compatible with index access


# Swapping out model providers

# In[6]:


lc_result = lc_openai.chat.completions.create(
    messages=messages, model="claude-2", temperature=0, provider="ChatAnthropic"
)
lc_result.choices[0].message


# ## chat.completions.stream

# Original OpenAI call

# In[7]:


for c in openai.chat.completions.create(
    messages=messages, model="gpt-3.5-turbo", temperature=0, stream=True
):
    print(c.choices[0].delta.model_dump())


# LangChain OpenAI wrapper call

# In[8]:


for c in lc_openai.chat.completions.create(
    messages=messages, model="gpt-3.5-turbo", temperature=0, stream=True
):
    print(c.choices[0].delta)


# Swapping out model providers

# In[9]:


for c in lc_openai.chat.completions.create(
    messages=messages,
    model="claude-2",
    temperature=0,
    stream=True,
    provider="ChatAnthropic",
):
    print(c["choices"][0]["delta"])
