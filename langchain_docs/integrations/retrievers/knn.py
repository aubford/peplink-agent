#!/usr/bin/env python
# coding: utf-8

# # kNN
#
# >In statistics, the [k-nearest neighbours algorithm (k-NN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) is a non-parametric supervised learning method first developed by `Evelyn Fix` and `Joseph Hodges` in 1951, and later expanded by `Thomas Cover`. It is used for classification and regression.
#
# This notebook goes over how to use a retriever that under the hood uses a kNN.
#
# Largely based on the code of [Andrej Karpathy](https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.html).

# In[ ]:


from langchain_community.retrievers import KNNRetriever
from langchain_openai import OpenAIEmbeddings


# ## Create New Retriever with Texts

# In[2]:


retriever = KNNRetriever.from_texts(
    ["foo", "bar", "world", "hello", "foo bar"], OpenAIEmbeddings()
)


# ## Use Retriever
#
# We can now use the retriever!

# In[3]:


result = retriever.invoke("foo")


# In[4]:


result
