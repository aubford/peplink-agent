#!/usr/bin/env python
# coding: utf-8

# # DuckDB
# This notebook shows how to use `DuckDB` as a vector store.

# In[ ]:


get_ipython().system(
    " pip install duckdb langchain langchain-community langchain-openai"
)


# We want to use OpenAIEmbeddings so we have to get the OpenAI API Key.

# In[2]:


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")


# In[ ]:


from langchain_community.vectorstores import DuckDB
from langchain_openai import OpenAIEmbeddings


# In[ ]:


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()

documents = CharacterTextSplitter().split_documents(documents)
embeddings = OpenAIEmbeddings()


# In[ ]:


docsearch = DuckDB.from_documents(documents, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)


# In[ ]:


print(docs[0].page_content)
