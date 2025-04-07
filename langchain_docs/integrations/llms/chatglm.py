#!/usr/bin/env python
# coding: utf-8

# # ChatGLM
# 
# [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) is an open bilingual language model based on General Language Model (GLM) framework, with 6.2 billion parameters. With the quantization technique, users can deploy locally on consumer-grade graphics cards (only 6GB of GPU memory is required at the INT4 quantization level). 
# 
# [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B) is the second-generation version of the open-source bilingual (Chinese-English) chat model ChatGLM-6B. It retains the smooth conversation flow and low deployment threshold of the first-generation model, while introducing the new features like better performance, longer context and more efficient inference.
# 
# [ChatGLM3](https://github.com/THUDM/ChatGLM3) is a new generation of pre-trained dialogue models jointly released by Zhipu AI and Tsinghua KEG. ChatGLM3-6B is the open-source model in the ChatGLM3 series

# In[ ]:


# Install required dependencies

get_ipython().run_line_magic('pip', 'install -qU langchain langchain-community')


# ## ChatGLM3
# 
# This examples goes over how to use LangChain to interact with ChatGLM3-6B Inference for text completion.

# In[1]:


from langchain.chains import LLMChain
from langchain_community.llms.chatglm3 import ChatGLM3
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate


# In[2]:


template = """{question}"""
prompt = PromptTemplate.from_template(template)


# In[3]:


endpoint_url = "http://127.0.0.1:8000/v1/chat/completions"

messages = [
    AIMessage(content="我将从美国到中国来旅游，出行前希望了解中国的城市"),
    AIMessage(content="欢迎问我任何问题。"),
]

llm = ChatGLM3(
    endpoint_url=endpoint_url,
    max_tokens=80000,
    prefix_messages=messages,
    top_p=0.9,
)


# In[4]:


llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "北京和上海两座城市有什么不同？"

llm_chain.run(question)


# ## ChatGLM and ChatGLM2
# 
# The following example shows how to use LangChain to interact with the ChatGLM2-6B Inference to complete text.
# ChatGLM-6B and ChatGLM2-6B has the same api specs, so this example should work with both.

# In[21]:


from langchain.chains import LLMChain
from langchain_community.llms import ChatGLM
from langchain_core.prompts import PromptTemplate

# import os


# In[22]:


template = """{question}"""
prompt = PromptTemplate.from_template(template)


# In[23]:


# default endpoint_url for a local deployed ChatGLM api server
endpoint_url = "http://127.0.0.1:8000"

# direct access endpoint in a proxied environment
# os.environ['NO_PROXY'] = '127.0.0.1'

llm = ChatGLM(
    endpoint_url=endpoint_url,
    max_token=80000,
    history=[
        ["我将从美国到中国来旅游，出行前希望了解中国的城市", "欢迎问我任何问题。"]
    ],
    top_p=0.9,
    model_kwargs={"sample_model_args": False},
)

# turn on with_history only when you want the LLM object to keep track of the conversation history
# and send the accumulated context to the backend model api, which make it stateful. By default it is stateless.
# llm.with_history = True


# In[24]:


llm_chain = LLMChain(prompt=prompt, llm=llm)


# In[25]:


question = "北京和上海两座城市有什么不同？"

llm_chain.run(question)

