#!/usr/bin/env python
# coding: utf-8

# # AwaDB
# >[AwaDB](https://github.com/awa-ai/awadb) is an AI Native database for the search and storage of embedding vectors used by LLM Applications.
# 
# You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration
# 
# This notebook shows how to use functionality related to the `AwaDB`.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  awadb')


# In[ ]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import AwaDB
from langchain_text_splitters import CharacterTextSplitter


# In[ ]:


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


# In[ ]:


db = AwaDB.from_documents(docs)
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)


# In[4]:


print(docs[0].page_content)


# ## Similarity search with score

# The returned distance score is between 0-1. 0 is dissimilar, 1 is the most similar

# In[ ]:


docs = db.similarity_search_with_score(query)


# In[4]:


print(docs[0])


# ## Restore the table created and added data before

# AwaDB automatically persists added document data.
# 
# If you can restore the table you created and added before, you can just do this as below:

# In[ ]:


import awadb

awadb_client = awadb.Client()
ret = awadb_client.Load("langchain_awadb")
if ret:
    print("awadb load table success")
else:
    print("awadb load table failed")

awadb load table success
