#!/usr/bin/env python
# coding: utf-8

# # OpenAI Adapter(Old)
# 
# **Please ensure OpenAI library is less than 1.0.0; otherwise, refer to the newer doc [OpenAI Adapter](/docs/integrations/adapters/openai/).**
# 
# A lot of people get started with OpenAI but want to explore other models. LangChain's integrations with many model providers make this easy to do so. While LangChain has it's own message and model APIs, we've also made it as easy as possible to explore other models by exposing an adapter to adapt LangChain models to the OpenAI api.
# 
# At the moment this only deals with output and does not return other information (token counts, stop reasons, etc).

# In[ ]:


import openai
from langchain_community.adapters import openai as lc_openai


# ## ChatCompletion.create

# In[29]:


messages = [{"role": "user", "content": "hi"}]


# Original OpenAI call

# In[15]:


result = openai.ChatCompletion.create(
    messages=messages, model="gpt-3.5-turbo", temperature=0
)
result["choices"][0]["message"].to_dict_recursive()


# LangChain OpenAI wrapper call

# In[17]:


lc_result = lc_openai.ChatCompletion.create(
    messages=messages, model="gpt-3.5-turbo", temperature=0
)
lc_result["choices"][0]["message"]


# Swapping out model providers

# In[19]:


lc_result = lc_openai.ChatCompletion.create(
    messages=messages, model="claude-2", temperature=0, provider="ChatAnthropic"
)
lc_result["choices"][0]["message"]


# ## ChatCompletion.stream

# Original OpenAI call

# In[24]:


for c in openai.ChatCompletion.create(
    messages=messages, model="gpt-3.5-turbo", temperature=0, stream=True
):
    print(c["choices"][0]["delta"].to_dict_recursive())


# LangChain OpenAI wrapper call

# In[30]:


for c in lc_openai.ChatCompletion.create(
    messages=messages, model="gpt-3.5-turbo", temperature=0, stream=True
):
    print(c["choices"][0]["delta"])


# Swapping out model providers

# In[31]:


for c in lc_openai.ChatCompletion.create(
    messages=messages,
    model="claude-2",
    temperature=0,
    stream=True,
    provider="ChatAnthropic",
):
    print(c["choices"][0]["delta"])

