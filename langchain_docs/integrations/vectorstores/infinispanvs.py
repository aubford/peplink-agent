#!/usr/bin/env python
# coding: utf-8

# # Infinispan
#
# Infinispan is an open-source key-value data grid, it can work as single node as well as distributed.
#
# Vector search is supported since release 15.x
# For more: [Infinispan Home](https://infinispan.org)

# In[ ]:


# Ensure that all we need is installed
# You may want to skip this
get_ipython().run_line_magic("pip", "install sentence-transformers")
get_ipython().run_line_magic("pip", "install langchain")
get_ipython().run_line_magic("pip", "install langchain_core")
get_ipython().run_line_magic("pip", "install langchain_community")


# # Setup
#
# To run this demo we need a running Infinispan instance without authentication and a data file.
# In the next three cells we're going to:
# - download the data file
# - create the configuration
# - run Infinispan in docker

# In[ ]:


get_ipython().run_cell_magic(
    "bash",
    "",
    "#get an archive of news\nwget https://raw.githubusercontent.com/rigazilla/infinispan-vector/main/bbc_news.csv.gz\n",
)


# In[ ]:


get_ipython().run_cell_magic(
    "bash",
    "",
    "#create infinispan configuration file\necho 'infinispan:\n  cache-container: \n    name: default\n    transport: \n      cluster: cluster \n      stack: tcp \n  server:\n    interfaces:\n      interface:\n        name: public\n        inet-address:\n          value: 0.0.0.0 \n    socket-bindings:\n      default-interface: public\n      port-offset: 0        \n      socket-binding:\n        name: default\n        port: 11222\n    endpoints:\n      endpoint:\n        socket-binding: default\n        rest-connector:\n' > infinispan-noauth.yaml\n",
)


# In[ ]:


get_ipython().system("docker rm --force infinispanvs-demo")
get_ipython().system(
    "docker run -d --name infinispanvs-demo -v $(pwd):/user-config  -p 11222:11222 infinispan/server:15.0 -c /user-config/infinispan-noauth.yaml"
)


# # The Code
#
# ## Pick up an embedding model
#
# In this demo we're using
# a HuggingFace embedding mode.

# In[ ]:


from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-MiniLM-L12-v2"
hf = HuggingFaceEmbeddings(model_name=model_name)


# ## Setup Infinispan cache
#
# Infinispan is a very flexible key-value store, it can store raw bits as well as complex data type.
# User has complete freedom in the datagrid configuration, but for simple data type everything is automatically
# configured by the python layer. We take advantage of this feature so we can focus on our application.

# ## Prepare the data
#
# In this demo we rely on the default configuration, thus texts, metadatas and vectors in the same cache, but other options are possible: i.e. content can be store somewhere else and vector store could contain only a reference to the actual content.

# In[ ]:


import csv
import gzip
import time

# Open the news file and process it as a csv
with gzip.open("bbc_news.csv.gz", "rt", newline="") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=",", quotechar='"')
    i = 0
    texts = []
    metas = []
    embeds = []
    for row in spamreader:
        # first and fifth values are joined to form the content
        # to be processed
        text = row[0] + "." + row[4]
        texts.append(text)
        # Store text and title as metadata
        meta = {"text": row[4], "title": row[0]}
        metas.append(meta)
        i = i + 1
        # Change this to change the number of news you want to load
        if i >= 5000:
            break


# # Populate the vector store

# In[ ]:


# add texts and fill vector db

from langchain_community.vectorstores import InfinispanVS

ispnvs = InfinispanVS.from_texts(texts, hf, metas)


# # An helper func that prints the result documents
#
# By default InfinispanVS returns the protobuf `ŧext` field in the `Document.page_content`
# and all the remaining protobuf fields (except the vector) in the `metadata`. This behaviour is
# configurable via lambda functions at setup.

# In[ ]:


def print_docs(docs):
    for res, i in zip(docs, range(len(docs))):
        print("----" + str(i + 1) + "----")
        print("TITLE: " + res.metadata["title"])
        print(res.page_content)


# # Try it!!!
#
# Below some sample queries

# In[ ]:


docs = ispnvs.similarity_search("European nations", 5)
print_docs(docs)


# In[ ]:


print_docs(ispnvs.similarity_search("Milan fashion week begins", 2))


# In[ ]:


print_docs(ispnvs.similarity_search("Stock market is rising today", 4))


# In[ ]:


print_docs(ispnvs.similarity_search("Why cats are so viral?", 2))


# In[ ]:


print_docs(ispnvs.similarity_search("How to stay young", 5))


# In[ ]:


get_ipython().system("docker rm --force infinispanvs-demo")
