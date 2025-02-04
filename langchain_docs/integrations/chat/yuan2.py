#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Yuan2.0
---
# # Yuan2.0
# 
# This notebook shows how to use [YUAN2 API](https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/docs/inference_server.md) in LangChain with the langchain.chat_models.ChatYuan2.
# 
# [*Yuan2.0*](https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/README-EN.md) is a new generation Fundamental Large Language Model developed by IEIT System. We have published all three models, Yuan 2.0-102B, Yuan 2.0-51B, and Yuan 2.0-2B. And we provide relevant scripts for pretraining, fine-tuning, and inference services for other developers. Yuan2.0 is based on Yuan1.0, utilizing a wider range of high-quality pre training data and instruction fine-tuning datasets to enhance the model's understanding of semantics, mathematics, reasoning, code, knowledge, and other aspects.

# ## Getting started
# ### Installation
# First, Yuan2.0 provided an OpenAI compatible API, and we integrate ChatYuan2 into langchain chat model by using OpenAI client.
# Therefore, ensure the openai package is installed in your Python environment. Run the following command:

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet openai')


# ### Importing the Required Modules
# After installation, import the necessary modules to your Python script:

# In[ ]:


from langchain_community.chat_models import ChatYuan2
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# ### Setting Up Your API server
# Setting up your OpenAI compatible API server following [yuan2 openai api server](https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/docs/Yuan2_fastchat.md).
# If you deployed api server locally, you can simply set `yuan2_api_key="EMPTY"` or anything you want.
# Just make sure, the `yuan2_api_base` is set correctly.

# In[ ]:


yuan2_api_key = "your_api_key"
yuan2_api_base = "http://127.0.0.1:8001/v1"


# ### Initialize the ChatYuan2 Model
# Here's how to initialize the chat model:

# In[ ]:


chat = ChatYuan2(
    yuan2_api_base="http://127.0.0.1:8001/v1",
    temperature=1.0,
    model_name="yuan2",
    max_retries=3,
    streaming=False,
)


# ### Basic Usage
# Invoke the model with system and human messages like this:

# In[ ]:


messages = [
    SystemMessage(content="你是一个人工智能助手。"),
    HumanMessage(content="你好，你是谁？"),
]


# In[ ]:


print(chat.invoke(messages))


# ### Basic Usage with streaming
# For continuous interaction, use the streaming feature:

# In[ ]:


from langchain_core.callbacks import StreamingStdOutCallbackHandler

chat = ChatYuan2(
    yuan2_api_base="http://127.0.0.1:8001/v1",
    temperature=1.0,
    model_name="yuan2",
    max_retries=3,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)
messages = [
    SystemMessage(content="你是个旅游小助手。"),
    HumanMessage(content="给我介绍一下北京有哪些好玩的。"),
]


# In[ ]:


chat.invoke(messages)


# ## Advanced Features
# ### Usage with async calls
# 
# Invoke the model with non-blocking calls, like this:

# In[ ]:


async def basic_agenerate():
    chat = ChatYuan2(
        yuan2_api_base="http://127.0.0.1:8001/v1",
        temperature=1.0,
        model_name="yuan2",
        max_retries=3,
    )
    messages = [
        [
            SystemMessage(content="你是个旅游小助手。"),
            HumanMessage(content="给我介绍一下北京有哪些好玩的。"),
        ]
    ]

    result = await chat.agenerate(messages)
    print(result)


# In[ ]:


import asyncio

asyncio.run(basic_agenerate())


# ### Usage with prompt template
# 
# Invoke the model with non-blocking calls and used chat template like this:

# In[ ]:


async def ainvoke_with_prompt_template():
    from langchain_core.prompts.chat import (
        ChatPromptTemplate,
    )

    chat = ChatYuan2(
        yuan2_api_base="http://127.0.0.1:8001/v1",
        temperature=1.0,
        model_name="yuan2",
        max_retries=3,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个诗人，擅长写诗。"),
            ("human", "给我写首诗，主题是{theme}。"),
        ]
    )
    chain = prompt | chat
    result = await chain.ainvoke({"theme": "明月"})
    print(f"type(result): {type(result)}; {result}")


# In[ ]:


asyncio.run(ainvoke_with_prompt_template())


# ### Usage with async calls in streaming
# For non-blocking calls with streaming output, use the astream method:

# In[ ]:


async def basic_astream():
    chat = ChatYuan2(
        yuan2_api_base="http://127.0.0.1:8001/v1",
        temperature=1.0,
        model_name="yuan2",
        max_retries=3,
    )
    messages = [
        SystemMessage(content="你是个旅游小助手。"),
        HumanMessage(content="给我介绍一下北京有哪些好玩的。"),
    ]
    result = chat.astream(messages)
    async for chunk in result:
        print(chunk.content, end="", flush=True)


# In[ ]:


import asyncio

asyncio.run(basic_astream())

