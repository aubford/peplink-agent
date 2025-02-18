#!/usr/bin/env python
# coding: utf-8

# # USearch
# >[USearch](https://unum-cloud.github.io/usearch/) is a Smaller & Faster Single-File Vector Search Engine
#
# >USearch's base functionality is identical to FAISS, and the interface should look familiar if you have ever investigated Approximate Nearest Neigbors search. FAISS is a widely recognized standard for high-performance vector search engines. USearch and FAISS both employ the same HNSW algorithm, but they differ significantly in their design principles. USearch is compact and broadly compatible without sacrificing performance, with a primary focus on user-defined metrics and fewer dependencies.

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet  usearch langchain-community"
)


# We want to use OpenAIEmbeddings so we have to get the OpenAI API Key.

# In[2]:


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")


# In[2]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import USearch
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


# In[3]:


from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../../extras/modules/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


# In[4]:


db = USearch.from_documents(docs, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)


# In[5]:


print(docs[0].page_content)


# ## Similarity Search with score
# The `similarity_search_with_score` method allows you to return not only the documents but also the distance score of the query to them. The returned distance score is L2 distance. Therefore, a lower score is better.

# In[6]:


docs_and_scores = db.similarity_search_with_score(query)


# In[7]:


docs_and_scores[0]


# In[ ]:
