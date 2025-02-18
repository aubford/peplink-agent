#!/usr/bin/env python
# coding: utf-8

# ## Fireworks.AI + LangChain + RAG
#
# [Fireworks AI](https://python.langchain.com/docs/integrations/llms/fireworks) wants to provide the best experience when working with LangChain, and here is an example of Fireworks + LangChain doing RAG
#
# See [our models page](https://fireworks.ai/models) for the full list of models. We use `accounts/fireworks/models/mixtral-8x7b-instruct` for RAG In this tutorial.
#
# For the RAG target, we will use the Gemma technical report https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf

# In[1]:


get_ipython().run_line_magic(
    "pip", "install --quiet pypdf langchain-chroma tiktoken openai"
)
get_ipython().run_line_magic("pip", "uninstall -y langchain-fireworks")
get_ipython().run_line_magic(
    "pip", "install --editable /mnt/disks/data/langchain/libs/partners/fireworks"
)


# In[3]:


import fireworks

print(fireworks)
import fireworks.client


# In[ ]:


# Load
import requests
from langchain_community.document_loaders import PyPDFLoader

# Download the PDF from a URL and save it to a temporary location
url = "https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf"
response = requests.get(url, stream=True)
file_name = "temp_file.pdf"
with open(file_name, "wb") as pdf:
    pdf.write(response.content)

loader = PyPDFLoader(file_name)
data = loader.load()

# Split
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Add to vectorDB
from langchain_chroma import Chroma
from langchain_fireworks.embeddings import FireworksEmbeddings

vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma",
    embedding=FireworksEmbeddings(),
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
