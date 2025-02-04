#!/usr/bin/env python
# coding: utf-8

# # RAG based on Qianfan and BES

# This notebook is an implementation of Retrieval augmented generation (RAG) using Baidu Qianfan Platform combined with Baidu ElasricSearch, where the original data is located on BOS.
# ## Baidu Qianfan
# Baidu AI Cloud Qianfan Platform is a one-stop large model development and service operation platform for enterprise developers. Qianfan not only provides including the model of Wenxin Yiyan (ERNIE-Bot) and the third-party open-source models, but also provides various AI development tools and the whole set of development environment, which facilitates customers to use and develop large model applications easily.
# 
# ## Baidu ElasticSearch
# [Baidu Cloud VectorSearch](https://cloud.baidu.com/doc/BES/index.html?from=productToDoc) is a fully managed, enterprise-level distributed search and analysis service which is 100% compatible to open source. Baidu Cloud VectorSearch provides low-cost, high-performance, and reliable retrieval and analysis platform level product services for structured/unstructured data. As a vector database , it supports multiple index types and similarity distance methods. 

# ## Installation and Setup
# 

# In[ ]:


#!pip install qianfan
#!pip install bce-python-sdk
#!pip install elasticsearch == 7.11.0
#!pip install sentence-transformers


# ## Imports

# In[ ]:


import sentence_transformers
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration
from langchain.chains.retrieval_qa import RetrievalQA
from langchain_community.document_loaders.baiducloud_bos_directory import (
    BaiduBOSDirectoryLoader,
)
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms.baidu_qianfan_endpoint import QianfanLLMEndpoint
from langchain_community.vectorstores import BESVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ## Document loading

# In[ ]:


bos_host = "your bos eddpoint"
access_key_id = "your bos access ak"
secret_access_key = "your bos access sk"

# create BceClientConfiguration
config = BceClientConfiguration(
    credentials=BceCredentials(access_key_id, secret_access_key), endpoint=bos_host
)

loader = BaiduBOSDirectoryLoader(conf=config, bucket="llm-test", prefix="llm/")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
split_docs = text_splitter.split_documents(documents)


# ## Embedding and VectorStore

# In[ ]:


embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
embeddings.client = sentence_transformers.SentenceTransformer(embeddings.model_name)

db = BESVectorStore.from_documents(
    documents=split_docs,
    embedding=embeddings,
    bes_url="your bes url",
    index_name="test-index",
    vector_query_field="vector",
)

db.client.indices.refresh(index="test-index")
retriever = db.as_retriever()


# ## QA Retriever

# In[ ]:


llm = QianfanLLMEndpoint(
    model="ERNIE-Bot",
    qianfan_ak="your qianfan ak",
    qianfan_sk="your qianfan sk",
    streaming=True,
)
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="refine", retriever=retriever, return_source_documents=True
)

query = "什么是张量?"
print(qa.run(query))


# > 张量（Tensor）是一个数学概念，用于表示多维数据。它是一个可以表示多个数值的数组，可以是标量、向量、矩阵等。在深度学习和人工智能领域中，张量常用于表示神经网络的输入、输出和权重等。
