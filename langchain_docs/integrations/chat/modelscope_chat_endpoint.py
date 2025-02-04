#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: ModelScope
---
# # ModelScopeChatEndpoint
# 
# 
# ModelScope ([Home](https://www.modelscope.cn/) | [GitHub](https://github.com/modelscope/modelscope)) is built upon the notion of “Model-as-a-Service” (MaaS). It seeks to bring together most advanced machine learning models from the AI community, and streamlines the process of leveraging AI models in real-world applications. The core ModelScope library open-sourced in this repository provides the interfaces and implementations that allow developers to perform model inference, training and evaluation. 
# 
# This will help you getting started with ModelScope Chat Endpoint.
# 
# 
# ## Overview
# ### Integration details
# 
# |Provider| Class | Package | Local | Serializable | Package downloads | Package latest |
# |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
# |[ModelScope](/docs/integrations/providers/modelscope/)| ModelScopeChatEndpoint | [langchain-modelscope-integration](https://pypi.org/project/langchain-modelscope-integration/) | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-modelscope-integration?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-modelscope-integration?style=flat-square&label=%20) |
# 
# 
# ## Setup
# 
# To access ModelScope chat endpoint you'll need to create a ModelScope account, get an SDK token, and install the `langchain-modelscope-integration` integration package.
# 
# ### Credentials
# 
# Head to [ModelScope](https://modelscope.cn/) to sign up to ModelScope and generate an [SDK token](https://modelscope.cn/my/myaccesstoken). Once you've done this set the `MODELSCOPE_SDK_TOKEN` environment variable:
# 

# In[1]:


import getpass
import os

if not os.getenv("MODELSCOPE_SDK_TOKEN"):
    os.environ["MODELSCOPE_SDK_TOKEN"] = getpass.getpass(
        "Enter your ModelScope SDK token: "
    )


# ### Installation
# 
# The LangChain ModelScope integration lives in the `langchain-modelscope-integration` package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-modelscope-integration')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:
# 

# In[3]:


from langchain_modelscope import ModelScopeChatEndpoint

llm = ModelScopeChatEndpoint(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0,
    max_tokens=1024,
    timeout=60,
    max_retries=2,
    # other params...
)


# ## Invocation
# 

# In[4]:


messages = [
    (
        "system",
        "You are a helpful assistant that translates English to Chinese. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg


# In[5]:


print(ai_msg.content)


# ## Chaining
# 
# We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:
# 

# In[6]:


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "input_language": "English",
        "output_language": "Chinese",
        "input": "I love programming.",
    }
)


# ## API reference
# 
# For detailed documentation of all ModelScopeChatEndpoint features and configurations head to the reference: https://modelscope.cn/docs/model-service/API-Inference/intro
# 
