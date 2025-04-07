#!/usr/bin/env python
# coding: utf-8

# # YandexGPT
# 
# This notebook goes over how to use Langchain with [YandexGPT](https://cloud.yandex.com/en/services/yandexgpt) embeddings models.
# 
# To use, you should have the `yandexcloud` python package installed.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  yandexcloud')


# First, you should [create service account](https://cloud.yandex.com/en/docs/iam/operations/sa/create) with the `ai.languageModels.user` role.
# 
# Next, you have two authentication options:
# - [IAM token](https://cloud.yandex.com/en/docs/iam/operations/iam-token/create-for-sa).
#     You can specify the token in a constructor parameter `iam_token` or in an environment variable `YC_IAM_TOKEN`.
# - [API key](https://cloud.yandex.com/en/docs/iam/operations/api-key/create)
#     You can specify the key in a constructor parameter `api_key` or in an environment variable `YC_API_KEY`.
# 
# To specify the model you can use `model_uri` parameter, see [the documentation](https://cloud.yandex.com/en/docs/yandexgpt/concepts/models#yandexgpt-embeddings) for more details.
# 
# By default, the latest version of `text-search-query` is used from the folder specified in the parameter `folder_id` or `YC_FOLDER_ID` environment variable.

# In[1]:


from langchain_community.embeddings.yandex import YandexGPTEmbeddings


# In[2]:


embeddings = YandexGPTEmbeddings()


# In[3]:


text = "This is a test document."


# In[4]:


query_result = embeddings.embed_query(text)


# In[5]:


doc_result = embeddings.embed_documents([text])


# In[6]:


query_result[:5]


# In[7]:


doc_result[0][:5]

