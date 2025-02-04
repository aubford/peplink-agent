#!/usr/bin/env python
# coding: utf-8

# # RankLLM Reranker
# 

# [RankLLM](https://github.com/castorini/rank_llm) offers a suite of listwise rerankers, albeit with focus on open source LLMs finetuned for the task - RankVicuna and RankZephyr being two of them.

# In[1]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  rank_llm')


# In[2]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  langchain_openai')


# In[3]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  faiss-cpu')


# In[4]:


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")


# In[5]:


# Helper function for printing docs
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# ## Set up the base vector store retriever
# Let's start by initializing a simple vector store retriever and storing the 2023 State of the Union speech (in chunks). We can set up the retriever to retrieve a high number (20) of docs.

# In[6]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = TextLoader("../../modules/state_of_the_union.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
for idx, text in enumerate(texts):
    text.metadata["id"] = idx

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
retriever = FAISS.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 20})


# # Retrieval + RankLLM Reranking (RankZephyr)

# Retrieval without reranking

# In[7]:


query = "What was done to Russia?"
docs = retriever.invoke(query)
pretty_print_docs(docs)


# Retrieval + Reranking with RankZephyr

# In[12]:


from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank

compressor = RankLLMRerank(top_n=3, model="zephyr")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


# In[9]:


compressed_docs = compression_retriever.invoke(query)
pretty_print_docs(compressed_docs)


# Can be used within a QA pipeline

# In[10]:


from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0), retriever=compression_retriever
)

chain({"query": query})


# # Retrieval + RankLLM Reranking (RankGPT)

# Retrieval without reranking

# In[11]:


query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.invoke(query)
pretty_print_docs(docs)


# Retrieval + Reranking with RankGPT

# In[12]:


from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank

compressor = RankLLMRerank(top_n=3, model="gpt", gpt_model="gpt-3.5-turbo")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


# In[13]:


compressed_docs = compression_retriever.invoke(query)
pretty_print_docs(compressed_docs)


# You can use this retriever within a QA pipeline

# In[14]:


from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0), retriever=compression_retriever
)

chain({"query": query})

