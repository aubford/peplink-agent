#!/usr/bin/env python
# coding: utf-8

# # StarRocks
# 
# >[StarRocks](https://www.starrocks.io/) is a High-Performance Analytical Database.
# `StarRocks` is a next-gen sub-second MPP database for full analytics scenarios, including multi-dimensional analytics, real-time analytics and ad-hoc query.
# 
# >Usually `StarRocks` is categorized into OLAP, and it has showed excellent performance in [ClickBench — a Benchmark For Analytical DBMS](https://benchmark.clickhouse.com/). Since it has a super-fast vectorized execution engine, it could also be used as a fast vectordb.
# 
# Here we'll show how to use the StarRocks Vector Store.

# ## Setup

# In[1]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  pymysql langchain-community')


# Set `update_vectordb = False` at the beginning. If there is no docs updated, then we don't need to rebuild the embeddings of docs

# In[1]:


from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import StarRocks
from langchain_community.vectorstores.starrocks import StarRocksSettings
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter

update_vectordb = False


# ## Load docs and split them into tokens

# Load all markdown files under the `docs` directory
# 
# for starrocks documents, you can clone repo from https://github.com/StarRocks/starrocks, and there is `docs` directory in it.

# In[2]:


loader = DirectoryLoader(
    "./docs", glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
)
documents = loader.load()


# Split docs into tokens, and set `update_vectordb = True` because there are new docs/tokens.

# In[3]:


# load text splitter and split docs into snippets of text
text_splitter = TokenTextSplitter(chunk_size=400, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

# tell vectordb to update text embeddings
update_vectordb = True


# In[4]:


split_docs[-20]


# In[5]:


print("# docs  = %d, # splits = %d" % (len(documents), len(split_docs)))


# ## Create vectordb instance

# ### Use StarRocks as vectordb

# In[6]:


def gen_starrocks(update_vectordb, embeddings, settings):
    if update_vectordb:
        docsearch = StarRocks.from_documents(split_docs, embeddings, config=settings)
    else:
        docsearch = StarRocks(embeddings, settings)
    return docsearch


# ## Convert tokens into embeddings and put them into vectordb

# Here we use StarRocks as vectordb, you can configure StarRocks instance via `StarRocksSettings`.
# 
# Configuring StarRocks instance is pretty much like configuring mysql instance. You need to specify:
# 1. host/port
# 2. username(default: 'root')
# 3. password(default: '')
# 4. database(default: 'default')
# 5. table(default: 'langchain')

# In[8]:


embeddings = OpenAIEmbeddings()

# configure starrocks settings(host/port/user/pw/db)
settings = StarRocksSettings()
settings.port = 41003
settings.host = "127.0.0.1"
settings.username = "root"
settings.password = ""
settings.database = "zya"
docsearch = gen_starrocks(update_vectordb, embeddings, settings)

print(docsearch)

update_vectordb = False


# ## Build QA and ask question to it

# In[10]:


llm = OpenAI()
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
)
query = "is profile enabled by default? if not, how to enable profile?"
resp = qa.run(query)
print(resp)

