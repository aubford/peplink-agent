#!/usr/bin/env python
# coding: utf-8

# # GigaChat
# This notebook shows how to use LangChain with [GigaChat](https://developers.sber.ru/portal/products/gigachat).
# To use you need to install ```gigachat``` python package.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  gigachat')


# To get GigaChat credentials you need to [create account](https://developers.sber.ru/studio/login) and [get access to API](https://developers.sber.ru/docs/ru/gigachat/individuals-quickstart)
# 
# ## Example

# In[2]:


import os
from getpass import getpass

if "GIGACHAT_CREDENTIALS" not in os.environ:
    os.environ["GIGACHAT_CREDENTIALS"] = getpass()


# In[3]:


from langchain_community.llms import GigaChat

llm = GigaChat(verify_ssl_certs=False, scope="GIGACHAT_API_PERS")


# In[9]:


from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

template = "What is capital of {country}?"

prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(prompt=prompt, llm=llm)

generated = llm_chain.invoke(input={"country": "Russia"})
print(generated["text"])

