#!/usr/bin/env python
# coding: utf-8

# # YandexGPT
# 
# This notebook goes over how to use Langchain with [YandexGPT](https://cloud.yandex.com/en/services/yandexgpt).
# 
# To use, you should have the `yandexcloud` python package installed.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  yandexcloud')


# First, you should [create service account](https://cloud.yandex.com/en/docs/iam/operations/sa/create) with the `ai.languageModels.user` role.
# 
# Next, you have two authentication options:
# - [IAM token](https://cloud.yandex.com/en/docs/iam/operations/iam-token/create-for-sa).
#     You can specify the token in a constructor parameter `iam_token` or in an environment variable `YC_IAM_TOKEN`.
# 
# - [API key](https://cloud.yandex.com/en/docs/iam/operations/api-key/create)
#     You can specify the key in a constructor parameter `api_key` or in an environment variable `YC_API_KEY`.
# 
# To specify the model you can use `model_uri` parameter, see [the documentation](https://cloud.yandex.com/en/docs/yandexgpt/concepts/models#yandexgpt-generation) for more details.
# 
# By default, the latest version of `yandexgpt-lite` is used from the folder specified in the parameter `folder_id` or `YC_FOLDER_ID` environment variable.

# In[1]:


from langchain.chains import LLMChain
from langchain_community.llms import YandexGPT
from langchain_core.prompts import PromptTemplate


# In[2]:


template = "What is the capital of {country}?"
prompt = PromptTemplate.from_template(template)


# In[3]:


llm = YandexGPT()


# In[4]:


llm_chain = LLMChain(prompt=prompt, llm=llm)


# In[5]:


country = "Russia"

llm_chain.invoke(country)

