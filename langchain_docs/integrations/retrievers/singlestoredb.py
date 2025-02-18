#!/usr/bin/env python
# coding: utf-8

# # SingleStoreDB
#
# >[SingleStoreDB](https://singlestore.com/) is a high-performance distributed SQL database that supports deployment both in the [cloud](https://www.singlestore.com/cloud/) and on-premises. It provides vector storage, and vector functions including [dot_product](https://docs.singlestore.com/managed-service/en/reference/sql-reference/vector-functions/dot_product.html) and [euclidean_distance](https://docs.singlestore.com/managed-service/en/reference/sql-reference/vector-functions/euclidean_distance.html), thereby supporting AI applications that require text similarity matching.
#
#
# This notebook shows how to use a retriever that uses `SingleStoreDB`.
#

# In[ ]:


# Establishing a connection to the database is facilitated through the singlestoredb Python connector.
# Please ensure that this connector is installed in your working environment.
get_ipython().run_line_magic("pip", "install --upgrade --quiet  singlestoredb")


# ## Create Retriever from vector store

# In[ ]:


import getpass
import os

# We want to use OpenAIEmbeddings so we have to get the OpenAI API Key.
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import SingleStoreDB
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

# Setup connection url as environment variable
os.environ["SINGLESTOREDB_URL"] = "root:pass@localhost:3306/db"

# Load documents to the store
docsearch = SingleStoreDB.from_documents(
    docs,
    embeddings,
    table_name="notebook",  # use table with a custom name
)

# create retriever from the vector store
retriever = docsearch.as_retriever(search_kwargs={"k": 2})


# ## Search with retriever

# In[13]:


result = retriever.invoke("What did the president say about Ketanji Brown Jackson")
print(docs[0].page_content)
