#!/usr/bin/env python
# coding: utf-8

# # Zilliz
#
# >[Zilliz Cloud](https://zilliz.com/doc/quick_start) is a fully managed service on cloud for `LF AI Milvus®`,
#
# This notebook shows how to use functionality related to the Zilliz Cloud managed vector database.
#
# You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration
#
# To run, you should have a `Zilliz Cloud` instance up and running. Here are the [installation instructions](https://zilliz.com/cloud)

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  pymilvus")


# We want to use `OpenAIEmbeddings` so we have to get the OpenAI API Key.

# In[1]:


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")


# In[2]:


# replace
ZILLIZ_CLOUD_URI = ""  # example: "https://in01-17f69c292d4a5sa.aws-us-west-2.vectordb.zillizcloud.com:19536"
ZILLIZ_CLOUD_USERNAME = ""  # example: "username"
ZILLIZ_CLOUD_PASSWORD = ""  # example: "*********"
ZILLIZ_CLOUD_API_KEY = ""  # example: "*********" (for serverless clusters which can be used as replacements for user and password)


# In[3]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Milvus
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


vector_db = Milvus.from_documents(
    docs,
    embeddings,
    connection_args={
        "uri": ZILLIZ_CLOUD_URI,
        "user": ZILLIZ_CLOUD_USERNAME,
        "password": ZILLIZ_CLOUD_PASSWORD,
        # "token": ZILLIZ_CLOUD_API_KEY,  # API key, for serverless clusters which can be used as replacements for user and password
        "secure": True,
    },
)


# In[6]:


query = "What did the president say about Ketanji Brown Jackson"
docs = vector_db.similarity_search(query)


# In[7]:


docs[0].page_content


# In[ ]:
