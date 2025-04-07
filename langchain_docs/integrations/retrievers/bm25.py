#!/usr/bin/env python
# coding: utf-8

# # BM25
# 
# >[BM25 (Wikipedia)](https://en.wikipedia.org/wiki/Okapi_BM25) also known as the `Okapi BM25`, is a ranking function used in information retrieval systems to estimate the relevance of documents to a given search query.
# >
# >`BM25Retriever` retriever uses the [`rank_bm25`](https://github.com/dorianbrown/rank_bm25) package.
# 

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  rank_bm25')


# In[3]:


from langchain_community.retrievers import BM25Retriever


# ## Create New Retriever with Texts

# In[4]:


retriever = BM25Retriever.from_texts(["foo", "bar", "world", "hello", "foo bar"])


# ## Create a New Retriever with Documents
# 
# You can now create a new retriever with the documents you created.

# In[5]:


from langchain_core.documents import Document

retriever = BM25Retriever.from_documents(
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


# ## Preprocessing Function
# Pass a custom preprocessing function to the retriever to improve search results. Tokenizing text at the word level can enhance retrieval, especially when using vector stores like Chroma, Pinecone, or Faiss for chunked documents.

# In[ ]:


import nltk

nltk.download("punkt_tab")


# In[32]:


from nltk.tokenize import word_tokenize

retriever = BM25Retriever.from_documents(
    [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="world"),
        Document(page_content="hello"),
        Document(page_content="foo bar"),
    ],
    k=2,
    preprocess_func=word_tokenize,
)

result = retriever.invoke("bar")
result

