#!/usr/bin/env python
# coding: utf-8

# # SVM
#
# >[Support vector machines (SVMs)](https://scikit-learn.org/stable/modules/svm.html#support-vector-machines) are a set of supervised learning methods used for classification, regression and outliers detection.
#
# This notebook goes over how to use a retriever that under the hood uses an `SVM` using `scikit-learn` package.
#
# Largely based on https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.html

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  scikit-learn")


# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  lark")


# We want to use `OpenAIEmbeddings` so we have to get the OpenAI API Key.

# In[4]:


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")


# In[5]:


from langchain_community.retrievers import SVMRetriever
from langchain_openai import OpenAIEmbeddings


# ## Create New Retriever with Texts

# In[6]:


retriever = SVMRetriever.from_texts(
    ["foo", "bar", "world", "hello", "foo bar"], OpenAIEmbeddings()
)


# ## Use Retriever
#
# We can now use the retriever!

# In[9]:


result = retriever.invoke("foo")


# In[10]:


result


# In[ ]:
