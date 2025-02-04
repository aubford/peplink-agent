#!/usr/bin/env python
# coding: utf-8

# # MyScale
# 
# >[MyScale](https://docs.myscale.com/en/overview/) is a cloud-based database optimized for AI applications and solutions, built on the open-source [ClickHouse](https://github.com/ClickHouse/ClickHouse). 
# 
# This notebook shows how to use functionality related to the `MyScale` vector database.

# ## Setting up environments

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  clickhouse-connect langchain-community')


# We want to use OpenAIEmbeddings so we have to get the OpenAI API Key.

# In[1]:


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
if "OPENAI_API_BASE" not in os.environ:
    os.environ["OPENAI_API_BASE"] = getpass.getpass("OpenAI Base:")
if "MYSCALE_HOST" not in os.environ:
    os.environ["MYSCALE_HOST"] = getpass.getpass("MyScale Host:")
if "MYSCALE_PORT" not in os.environ:
    os.environ["MYSCALE_PORT"] = getpass.getpass("MyScale Port:")
if "MYSCALE_USERNAME" not in os.environ:
    os.environ["MYSCALE_USERNAME"] = getpass.getpass("MyScale Username:")
if "MYSCALE_PASSWORD" not in os.environ:
    os.environ["MYSCALE_PASSWORD"] = getpass.getpass("MyScale Password:")


# There are two ways to set up parameters for myscale index.
# 
# 1. Environment Variables
# 
#     Before you run the app, please set the environment variable with `export`:
#     `export MYSCALE_HOST='<your-endpoints-url>' MYSCALE_PORT=<your-endpoints-port> MYSCALE_USERNAME=<your-username> MYSCALE_PASSWORD=<your-password> ...`
# 
#     You can easily find your account, password and other info on our SaaS. For details please refer to [this document](https://docs.myscale.com/en/cluster-management/)
# 
#     Every attributes under `MyScaleSettings` can be set with prefix `MYSCALE_` and is case insensitive.
# 
# 2. Create `MyScaleSettings` object with parameters
# 
# 
#     ```python
#     from langchain_community.vectorstores import MyScale, MyScaleSettings
#     config = MyScaleSetting(host="<your-backend-url>", port=8443, ...)
#     index = MyScale(embedding_function, config)
#     index.add_documents(...)
#     ```

# In[3]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import MyScale
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


# In[4]:


from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


# In[5]:


for d in docs:
    d.metadata = {"some": "metadata"}
docsearch = MyScale.from_documents(docs, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)


# In[7]:


print(docs[0].page_content)


# ## Get connection info and data schema

# In[ ]:


print(str(docsearch))


# ## Filtering
# 
# You can have direct access to myscale SQL where statement. You can write `WHERE` clause following standard SQL.
# 
# **NOTE**: Please be aware of SQL injection, this interface must not be directly called by end-user.
# 
# If you customized your `column_map` under your setting, you search with filter like this:

# In[7]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import MyScale

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

for i, d in enumerate(docs):
    d.metadata = {"doc_id": i}

docsearch = MyScale.from_documents(docs, embeddings)


# ### Similarity search with score

# The returned distance score is cosine distance. Therefore, a lower score is better.

# In[8]:


meta = docsearch.metadata_column
output = docsearch.similarity_search_with_relevance_scores(
    "What did the president say about Ketanji Brown Jackson?",
    k=4,
    where_str=f"{meta}.doc_id<10",
)
for d, dist in output:
    print(dist, d.metadata, d.page_content[:20] + "...")


# ## Deleting your data
# 
# You can either drop the table with `.drop()` method or partially delete your data with `.delete()` method.

# In[9]:


# use directly a `where_str` to delete
docsearch.delete(where_str=f"{docsearch.metadata_column}.doc_id < 5")
meta = docsearch.metadata_column
output = docsearch.similarity_search_with_relevance_scores(
    "What did the president say about Ketanji Brown Jackson?",
    k=4,
    where_str=f"{meta}.doc_id<10",
)
for d, dist in output:
    print(dist, d.metadata, d.page_content[:20] + "...")


# In[10]:


docsearch.drop()


# In[ ]:




