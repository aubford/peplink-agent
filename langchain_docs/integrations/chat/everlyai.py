#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: EverlyAI
---
# # ChatEverlyAI
# 
# >[EverlyAI](https://everlyai.xyz) allows you to run your ML models at scale in the cloud. It also provides API access to [several LLM models](https://everlyai.xyz).
# 
# This notebook demonstrates the use of `langchain.chat_models.ChatEverlyAI` for [EverlyAI Hosted Endpoints](https://everlyai.xyz/).
# 
# * Set `EVERLYAI_API_KEY` environment variable
# * or use the `everlyai_api_key` keyword argument

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  langchain-openai')


# In[1]:


import os
from getpass import getpass

if "EVERLYAI_API_KEY" not in os.environ:
    os.environ["EVERLYAI_API_KEY"] = getpass()


# # Let's try out LLAMA model offered on EverlyAI Hosted Endpoints

# In[2]:


from langchain_community.chat_models import ChatEverlyAI
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful AI that shares everything you know."),
    HumanMessage(
        content="Tell me technical facts about yourself. Are you a transformer model? How many billions of parameters do you have?"
    ),
]

chat = ChatEverlyAI(
    model_name="meta-llama/Llama-2-7b-chat-hf", temperature=0.3, max_tokens=64
)
print(chat(messages).content)


# # EverlyAI also supports streaming responses

# In[3]:


from langchain_community.chat_models import ChatEverlyAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a humorous AI that delights people."),
    HumanMessage(content="Tell me a joke?"),
]

chat = ChatEverlyAI(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    temperature=0.3,
    max_tokens=64,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)
chat(messages)


# # Let's try a different language model on EverlyAI

# In[4]:


from langchain_community.chat_models import ChatEverlyAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a humorous AI that delights people."),
    HumanMessage(content="Tell me a joke?"),
]

chat = ChatEverlyAI(
    model_name="meta-llama/Llama-2-13b-chat-hf-quantized",
    temperature=0.3,
    max_tokens=128,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)
chat(messages)

