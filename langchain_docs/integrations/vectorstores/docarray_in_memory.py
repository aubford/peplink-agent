#!/usr/bin/env python
# coding: utf-8

# # DocArray InMemorySearch
#
# >[DocArrayInMemorySearch](https://docs.docarray.org/user_guide/storing/index_in_memory/) is a document index provided by [Docarray](https://github.com/docarray/docarray) that stores documents in memory. It is a great starting point for small datasets, where you may not want to launch a database server.
#
# This notebook shows how to use functionality related to the `DocArrayInMemorySearch`.

# ## Setup
#
# Uncomment the below cells to install docarray and get/set your OpenAI api key if you haven't already done so.

# In[ ]:


get_ipython().run_line_magic(
    "pip", 'install --upgrade --quiet  langchain-community "docarray"'
)


# In[ ]:


# Get an OpenAI token: https://platform.openai.com/account/api-keys

# import os
# from getpass import getpass

# OPENAI_API_KEY = getpass()

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ## Using DocArrayInMemorySearch

# In[ ]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


# In[4]:


documents = TextLoader("../../how_to/state_of_the_union.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = DocArrayInMemorySearch.from_documents(docs, embeddings)


# ### Similarity search

# In[5]:


query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)


# In[6]:


print(docs[0].page_content)


# ### Similarity search with score

# The returned distance score is cosine distance. Therefore, a lower score is better.

# In[7]:


docs = db.similarity_search_with_score(query)


# In[8]:


docs[0]


# In[ ]:
