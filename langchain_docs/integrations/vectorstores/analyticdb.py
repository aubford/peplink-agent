#!/usr/bin/env python
# coding: utf-8

# # AnalyticDB
#
# >[AnalyticDB for PostgreSQL](https://www.alibabacloud.com/help/en/analyticdb-for-postgresql/latest/product-introduction-overview) is a massively parallel processing (MPP) data warehousing service that is designed to analyze large volumes of data online.
#
# >`AnalyticDB for PostgreSQL` is developed based on the open-source `Greenplum Database` project and is enhanced with in-depth extensions by `Alibaba Cloud`. AnalyticDB for PostgreSQL is compatible with the ANSI SQL 2003 syntax and the PostgreSQL and Oracle database ecosystems. AnalyticDB for PostgreSQL also supports row store and column store. AnalyticDB for PostgreSQL processes petabytes of data offline at a high performance level and supports highly concurrent online queries.
#
# You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration
#
# This notebook shows how to use functionality related to the `AnalyticDB` vector database.
# To run, you should have an [AnalyticDB](https://www.alibabacloud.com/help/en/analyticdb-for-postgresql/latest/product-introduction-overview) instance up and running:
#
# - Using [AnalyticDB Cloud Vector Database](https://www.alibabacloud.com/product/hybriddb-postgresql). Click here to fast deploy it.

# In[ ]:


from langchain_community.vectorstores import AnalyticDB
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


# Split documents and get embeddings by call OpenAI API

# In[2]:


from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


# Connect to AnalyticDB by setting related ENVIRONMENTS.
# ```
# export PG_HOST={your_analyticdb_hostname}
# export PG_PORT={your_analyticdb_port} # Optional, default is 5432
# export PG_DATABASE={your_database} # Optional, default is postgres
# export PG_USER={database_username}
# export PG_PASSWORD={database_password}
# ```
#
# Then store your embeddings and documents into AnalyticDB

# In[3]:


import os

connection_string = AnalyticDB.connection_string_from_db_params(
    driver=os.environ.get("PG_DRIVER", "psycopg2cffi"),
    host=os.environ.get("PG_HOST", "localhost"),
    port=int(os.environ.get("PG_PORT", "5432")),
    database=os.environ.get("PG_DATABASE", "postgres"),
    user=os.environ.get("PG_USER", "postgres"),
    password=os.environ.get("PG_PASSWORD", "postgres"),
)

vector_db = AnalyticDB.from_documents(
    docs,
    embeddings,
    connection_string=connection_string,
)


# Query and retrieve data

# In[4]:


query = "What did the president say about Ketanji Brown Jackson"
docs = vector_db.similarity_search(query)


# In[5]:


print(docs[0].page_content)
