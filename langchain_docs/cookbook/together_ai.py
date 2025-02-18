#!/usr/bin/env python
# coding: utf-8

# ## Together AI + RAG
#
# [Together AI](https://python.langchain.com/docs/integrations/llms/together) has a broad set of OSS LLMs via inference API.
#
# See [here](https://docs.together.ai/docs/inference-models). We use `"mistralai/Mixtral-8x7B-Instruct-v0.1` for RAG on the Mixtral paper.
#
# Download the paper:
# https://arxiv.org/pdf/2401.04088.pdf

# In[ ]:


get_ipython().system(
    " pip install --quiet pypdf tiktoken openai langchain-chroma langchain-together"
)


# In[ ]:


# Load
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("~/Desktop/mixtral.pdf")
data = loader.load()

# Split
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Add to vectorDB
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

"""
from langchain_together.embeddings import TogetherEmbeddings
embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
"""
vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)

retriever = vectorstore.as_retriever()


# In[3]:


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# RAG prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
from langchain_together import Together

llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.0,
    max_tokens=2000,
    top_k=1,
)

# RAG chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)


# In[4]:


chain.invoke("What are the Architectural details of Mixtral?")


# Trace:
#
# https://smith.langchain.com/public/935fd642-06a6-4b42-98e3-6074f93115cd/r
