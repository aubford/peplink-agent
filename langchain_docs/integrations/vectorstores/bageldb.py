#!/usr/bin/env python
# coding: utf-8

# # BagelDB
#
# > [BagelDB](https://www.bageldb.ai/) (`Open Vector Database for AI`), is like GitHub for AI data.
# It is a collaborative platform where users can create,
# share, and manage vector datasets. It can support private projects for independent developers,
# internal collaborations for enterprises, and public contributions for data DAOs.
#
# ### Installation and Setup
#
# ```bash
# pip install betabageldb langchain-community
# ```
#
#

# ## Create VectorStore from texts

# In[9]:


from langchain_community.vectorstores import Bagel

texts = ["hello bagel", "hello langchain", "I love salad", "my car", "a dog"]
# create cluster and add texts
cluster = Bagel.from_texts(cluster_name="testing", texts=texts)


# In[11]:


# similarity search
cluster.similarity_search("bagel", k=3)


# In[12]:


# the score is a distance metric, so lower is better
cluster.similarity_search_with_score("bagel", k=3)


# In[13]:


# delete the cluster
cluster.delete_cluster()


# ## Create VectorStore from docs

# In[33]:


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)[:10]


# In[36]:


# create cluster with docs
cluster = Bagel.from_documents(cluster_name="testing_with_docs", documents=docs)


# In[37]:


# similarity search
query = "What did the president say about Ketanji Brown Jackson"
docs = cluster.similarity_search(query)
print(docs[0].page_content[:102])


# ## Get all text/doc from Cluster

# In[53]:


texts = ["hello bagel", "this is langchain"]
cluster = Bagel.from_texts(cluster_name="testing", texts=texts)
cluster_data = cluster.get()


# In[54]:


# all keys
cluster_data.keys()


# In[56]:


# all values and keys
cluster_data


# In[57]:


cluster.delete_cluster()


# ## Create cluster with metadata & filter using metadata

# In[63]:


texts = ["hello bagel", "this is langchain"]
metadatas = [{"source": "notion"}, {"source": "google"}]

cluster = Bagel.from_texts(cluster_name="testing", texts=texts, metadatas=metadatas)
cluster.similarity_search_with_score("hello bagel", where={"source": "notion"})


# In[64]:


# delete the cluster
cluster.delete_cluster()
