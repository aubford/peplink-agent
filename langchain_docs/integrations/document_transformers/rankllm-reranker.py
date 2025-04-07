#!/usr/bin/env python
# coding: utf-8

# # RankLLM Reranker
# 

# **[RankLLM](https://github.com/castorini/rank_llm)** is a **flexible reranking framework** supporting **listwise, pairwise, and pointwise ranking models**. It includes **RankVicuna, RankZephyr, MonoT5, DuoT5, LiT5, and FirstMistral**, with integration for **FastChat, vLLM, SGLang, and TensorRT-LLM** for efficient inference. RankLLM is optimized for **retrieval and ranking tasks**, leveraging both **open-source LLMs** and proprietary rerankers like **RankGPT and RankGemini**. It supports **batched inference, first-token reranking, and retrieval via BM25 and SPLADE**.
# 
# > **Note:** If using the built-in retriever, RankLLM requires **Pyserini, JDK 21, PyTorch, and Faiss** for retrieval functionality.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet rank_llm')


# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet langchain_openai')


# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet faiss-cpu')


# In[ ]:


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")


# In[ ]:


# Helper function for printing docs
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# ## Set up the base vector store retriever
# Let's start by initializing a simple vector store retriever and storing the 2023 State of the Union speech (in chunks). We can set up the retriever to retrieve a high number (20) of docs.

# In[ ]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = TextLoader("../document_loaders/example_data/state_of_the_union.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
for idx, text in enumerate(texts):
    text.metadata["id"] = idx

embedding = OpenAIEmbeddings(model="text-embedding-3-large")
retriever = FAISS.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 20})


# # Retrieval + RankLLM Reranking (RankZephyr)

# Retrieval without reranking

# In[ ]:


query = "What was done to Russia?"
docs = retriever.invoke(query)
pretty_print_docs(docs)


# RankZephyr performs listwise reranking for improved retrieval quality but requires at least 24GB of VRAM to run efficiently.

# In[ ]:


import torch
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank

torch.cuda.empty_cache()

compressor = RankLLMRerank(top_n=3, model="rank_zephyr")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

del compressor


# In[ ]:


compressed_docs = compression_retriever.invoke(query)
pretty_print_docs(compressed_docs)


# Can be used within a QA pipeline

# In[ ]:


from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0), retriever=compression_retriever
)

chain.invoke({"query": query})


# # Retrieval + RankLLM Reranking (RankGPT)

# Retrieval without reranking

# In[ ]:


query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.invoke(query)
pretty_print_docs(docs)


# Retrieval + Reranking with RankGPT

# In[ ]:


from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank

compressor = RankLLMRerank(top_n=3, model="gpt", gpt_model="gpt-4o-mini")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


# In[ ]:


compressed_docs = compression_retriever.invoke(query)
pretty_print_docs(compressed_docs)


# You can use this retriever within a QA pipeline

# In[ ]:


from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0), retriever=compression_retriever
)

chain.invoke({"query": query})

