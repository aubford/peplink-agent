#!/usr/bin/env python
# coding: utf-8

# # SparkLLM
# [SparkLLM](https://xinghuo.xfyun.cn/spark) is a large-scale cognitive model independently developed by iFLYTEK.
# It has cross-domain knowledge and language understanding ability by learning a large amount of texts, codes and images.
# It can understand and perform tasks based on natural dialogue.

# ## Prerequisite
# - Get SparkLLM's app_id, api_key and api_secret from [iFlyTek SparkLLM API Console](https://console.xfyun.cn/services/bm3) (for more info, see [iFlyTek SparkLLM Intro](https://xinghuo.xfyun.cn/sparkapi) ), then set environment variables `IFLYTEK_SPARK_APP_ID`, `IFLYTEK_SPARK_API_KEY` and `IFLYTEK_SPARK_API_SECRET` or pass parameters when creating `ChatSparkLLM` as the demo above.

# ## Use SparkLLM

# In[1]:


import os

os.environ["IFLYTEK_SPARK_APP_ID"] = "app_id"
os.environ["IFLYTEK_SPARK_API_KEY"] = "api_key"
os.environ["IFLYTEK_SPARK_API_SECRET"] = "api_secret"


# In[2]:


from langchain_community.llms import SparkLLM

# Load the model
llm = SparkLLM()

res = llm.invoke("What's your name?")
print(res)


# In[9]:


res = llm.generate(prompts=["hello!"])
res


# In[8]:


for res in llm.stream("foo:"):
    print(res)


# In[ ]:




