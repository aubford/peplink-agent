#!/usr/bin/env python
# coding: utf-8

# # Baichuan LLM
# Baichuan Inc. (https://www.baichuan-ai.com/) is a Chinese startup in the era of AGI, dedicated to addressing fundamental human needs: Efficiency, Health, and Happiness.

# In[ ]:


##Installing the langchain packages needed to use the integration
get_ipython().run_line_magic("pip", "install -qU langchain-community")


# ## Prerequisite
# An API key is required to access Baichuan LLM API. Visit https://platform.baichuan-ai.com/ to get your API key.

# ## Use Baichuan LLM

# In[ ]:


import os

os.environ["BAICHUAN_API_KEY"] = "YOUR_API_KEY"


# In[ ]:


from langchain_community.llms import BaichuanLLM

# Load the model
llm = BaichuanLLM()

res = llm.invoke("What's your name?")
print(res)


# In[ ]:


res = llm.generate(prompts=["你好！"])
res


# In[ ]:


for res in llm.stream("Who won the second world war?"):
    print(res)


# In[ ]:


import asyncio


async def run_aio_stream():
    async for res in llm.astream("Write a poem about the sun."):
        print(res)


asyncio.run(run_aio_stream())
