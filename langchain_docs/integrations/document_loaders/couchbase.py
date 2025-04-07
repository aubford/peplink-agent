#!/usr/bin/env python
# coding: utf-8

# # Couchbase
# 
# >[Couchbase](http://couchbase.com/) is an award-winning distributed NoSQL cloud database that delivers unmatched versatility, performance, scalability, and financial value for all of your cloud, mobile, AI, and edge computing applications.
# 

# ## Installation

# In[1]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  couchbase')


# ## Querying for Documents from Couchbase
# For more details on connecting to a Couchbase cluster, please check the [Python SDK documentation](https://docs.couchbase.com/python-sdk/current/howtos/managing-connections.html#connection-strings).
# 
# For help with querying for documents using SQL++ (SQL for JSON), please check the [documentation](https://docs.couchbase.com/server/current/n1ql/n1ql-language-reference/index.html).

# In[2]:


from langchain_community.document_loaders.couchbase import CouchbaseLoader

connection_string = "couchbase://localhost"  # valid Couchbase connection string
db_username = (
    "Administrator"  # valid database user with read access to the bucket being queried
)
db_password = "Password"  # password for the database user

# query is a valid SQL++ query
query = """
    SELECT h.* FROM `travel-sample`.inventory.hotel h 
        WHERE h.country = 'United States'
        LIMIT 1
        """


# ## Create the Loader

# In[3]:


loader = CouchbaseLoader(
    connection_string,
    db_username,
    db_password,
    query,
)


# You can fetch the documents by calling the `load` method of the loader. It will return a list with all the documents. If you want to avoid this blocking call, you can call `lazy_load` method that returns an Iterator.

# In[4]:


docs = loader.load()
print(docs)


# In[5]:


docs_iterator = loader.lazy_load()
for doc in docs_iterator:
    print(doc)
    break


# ## Specifying Fields with Content and Metadata
# The fields that are part of the Document content can be specified using the `page_content_fields` parameter.
# The metadata fields for the Document can be specified using the `metadata_fields` parameter.

# In[6]:


loader_with_selected_fields = CouchbaseLoader(
    connection_string,
    db_username,
    db_password,
    query,
    page_content_fields=[
        "address",
        "name",
        "city",
        "phone",
        "country",
        "geo",
        "description",
        "reviews",
    ],
    metadata_fields=["id"],
)
docs_with_selected_fields = loader_with_selected_fields.load()
print(docs_with_selected_fields)

