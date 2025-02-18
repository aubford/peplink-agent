#!/usr/bin/env python
# coding: utf-8

# # Tablestore
#
# [Tablestore](https://www.aliyun.com/product/ots) is a fully managed NoSQL cloud database service.
#
# Tablestore enables storage of a massive amount of structured and semi-structured data.
#
# This notebook shows how to use functionality related to the `Tablestore` vector database.
#
# To use Tablestore, you must create an instance.
# Here are the [creating instance instructions](https://help.aliyun.com/zh/tablestore/getting-started/manage-the-wide-column-model-in-the-tablestore-console).

# ## Setup

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet  langchain-community tablestore"
)


# ## Initialization

# In[1]:


import getpass
import os

os.environ["end_point"] = getpass.getpass("Tablestore end_point:")
os.environ["instance_name"] = getpass.getpass("Tablestore instance_name:")
os.environ["access_key_id"] = getpass.getpass("Tablestore access_key_id:")
os.environ["access_key_secret"] = getpass.getpass("Tablestore access_key_secret:")


# Create vector store.

# In[2]:


import tablestore
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import TablestoreVectorStore
from langchain_core.documents import Document

test_embedding_dimension_size = 4
embeddings = FakeEmbeddings(size=test_embedding_dimension_size)

store = TablestoreVectorStore(
    embedding=embeddings,
    endpoint=os.getenv("end_point"),
    instance_name=os.getenv("instance_name"),
    access_key_id=os.getenv("access_key_id"),
    access_key_secret=os.getenv("access_key_secret"),
    vector_dimension=test_embedding_dimension_size,
    # metadata mapping is used to filter non-vector fields.
    metadata_mappings=[
        tablestore.FieldSchema(
            "type", tablestore.FieldType.KEYWORD, index=True, enable_sort_and_agg=True
        ),
        tablestore.FieldSchema(
            "time", tablestore.FieldType.LONG, index=True, enable_sort_and_agg=True
        ),
    ],
)


# ## Manage vector store

# Create table and index.

# In[3]:


store.create_table_if_not_exist()
store.create_search_index_if_not_exist()


# Add documents.

# In[4]:


store.add_documents(
    [
        Document(
            id="1", page_content="1 hello world", metadata={"type": "pc", "time": 2000}
        ),
        Document(
            id="2", page_content="abc world", metadata={"type": "pc", "time": 2009}
        ),
        Document(
            id="3", page_content="3 text world", metadata={"type": "sky", "time": 2010}
        ),
        Document(
            id="4", page_content="hi world", metadata={"type": "sky", "time": 2030}
        ),
        Document(
            id="5", page_content="hi world", metadata={"type": "sky", "time": 2030}
        ),
    ]
)


# Delete document.

# In[5]:


store.delete(["3"])


# Get documents.

# ## Query vector store

# In[6]:


store.get_by_ids(["1", "3", "5"])


# Similarity search.

# In[7]:


store.similarity_search(query="hello world", k=2)


# Similarity search with filters.

# In[8]:


store.similarity_search(
    query="hello world",
    k=10,
    tablestore_filter_query=tablestore.BoolQuery(
        must_queries=[tablestore.TermQuery(field_name="type", column_value="sky")],
        should_queries=[tablestore.RangeQuery(field_name="time", range_from=2020)],
        must_not_queries=[tablestore.TermQuery(field_name="type", column_value="pc")],
    ),
)


# ## Usage for retrieval-augmented generation
#
# For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:
#
# - [Tutorials](/docs/tutorials/)
# - [How-to: Question and answer with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
# - [Retrieval conceptual docs](https://python.langchain.com/docs/concepts/retrieval)

# ## API reference
#
# For detailed documentation of all `TablestoreVectorStore` features and configurations head to the API reference:
#  https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.tablestore.TablestoreVectorStore.html
