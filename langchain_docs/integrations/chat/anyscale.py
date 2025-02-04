#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Anyscale
---
# # ChatAnyscale
# 
# This notebook demonstrates the use of `langchain.chat_models.ChatAnyscale` for [Anyscale Endpoints](https://endpoints.anyscale.com/).
# 
# * Set `ANYSCALE_API_KEY` environment variable
# * or use the `anyscale_api_key` keyword argument

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  langchain-openai')


# In[1]:


import os
from getpass import getpass

if "ANYSCALE_API_KEY" not in os.environ:
    os.environ["ANYSCALE_API_KEY"] = getpass()


# # Let's try out each model offered on Anyscale Endpoints

# In[2]:


from langchain_community.chat_models import ChatAnyscale

chats = {
    model: ChatAnyscale(model_name=model, temperature=1.0)
    for model in ChatAnyscale.get_available_models()
}

print(chats.keys())


# # We can use async methods and other stuff supported by ChatOpenAI
# 
# This way, the three requests will only take as long as the longest individual request.

# In[3]:


import asyncio

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful AI that shares everything you know."),
    HumanMessage(
        content="Tell me technical facts about yourself. Are you a transformer model? How many billions of parameters do you have?"
    ),
]


async def get_msgs():
    tasks = [chat.apredict_messages(messages) for chat in chats.values()]
    responses = await asyncio.gather(*tasks)
    return dict(zip(chats.keys(), responses))


# In[4]:


import nest_asyncio

nest_asyncio.apply()


# In[5]:


get_ipython().run_cell_magic('time', '', '\nresponse_dict = asyncio.run(get_msgs())\n\nfor model_name, response in response_dict.items():\n    print(f"\\t{model_name}")\n    print()\n    print(response.content)\n    print("\\n---\\n")\n')

