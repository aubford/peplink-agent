#!/usr/bin/env python
# coding: utf-8

# # TF-IDF
#
# >[TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting) means term-frequency times inverse document-frequency.
#
# This notebook goes over how to use a retriever that under the hood uses [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) using `scikit-learn` package.
#
# For more information on the details of TF-IDF see [this blog post](https://medium.com/data-science-bootcamp/tf-idf-basics-of-information-retrieval-48de122b2a4c).

# In[2]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  scikit-learn")


# In[3]:


from langchain_community.retrievers import TFIDFRetriever


# ## Create New Retriever with Texts

# In[4]:


retriever = TFIDFRetriever.from_texts(["foo", "bar", "world", "hello", "foo bar"])


# ## Create a New Retriever with Documents
#
# You can now create a new retriever with the documents you created.

# In[5]:


from langchain_core.documents import Document

retriever = TFIDFRetriever.from_documents(
    [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="world"),
        Document(page_content="hello"),
        Document(page_content="foo bar"),
    ]
)


# ## Use Retriever
#
# We can now use the retriever!

# In[6]:


result = retriever.invoke("foo")


# In[7]:


result


# ## Save and load
#
# You can easily save and load this retriever, making it handy for local development!

# In[8]:


retriever.save_local("testing.pkl")


# In[9]:


retriever_copy = TFIDFRetriever.load_local("testing.pkl")


# In[10]:


retriever_copy.invoke("foo")


# In[ ]:
