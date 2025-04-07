#!/usr/bin/env python
# coding: utf-8

# ## Needle Retriever
# [Needle](https://needle-ai.com) makes it easy to create your RAG pipelines with minimal effort. 
# 
# For more details, refer to our [API documentation](https://docs.needle-ai.com/docs/api-reference/needle-api)

# ## Overview
# The Needle Document Loader is a utility for integrating Needle collections with LangChain. It enables seamless storage, retrieval, and utilization of documents for Retrieval-Augmented Generation (RAG) workflows.
# 
# This example demonstrates:
# 
# * Storing documents into a Needle collection.
# * Setting up a retriever to fetch documents.
# * Building a Retrieval-Augmented Generation (RAG) pipeline.

# ### Setup
# Before starting, ensure you have the following environment variables set:
# 
# * NEEDLE_API_KEY: Your API key for authenticating with Needle.
# * OPENAI_API_KEY: Your OpenAI API key for language model operations.

# ## Initialization
# To initialize the NeedleLoader, you need the following parameters:
# 
# * needle_api_key: Your Needle API key (or set it as an environment variable).
# * collection_id: The ID of the Needle collection to work with.

# In[1]:


import os


# In[2]:


os.environ["NEEDLE_API_KEY"] = ""


# In[3]:


os.environ["OPENAI_API_KEY"] = ""


# ## Instantiation

# In[ ]:


from langchain_community.document_loaders.needle import NeedleLoader

collection_id = "clt_01J87M9T6B71DHZTHNXYZQRG5H"

# Initialize NeedleLoader to store documents to the collection
document_loader = NeedleLoader(
    needle_api_key=os.getenv("NEEDLE_API_KEY"),
    collection_id=collection_id,
)


# ## Load
# To add files to the Needle collection:

# In[ ]:


files = {
    "tech-radar-30.pdf": "https://www.thoughtworks.com/content/dam/thoughtworks/documents/radar/2024/04/tr_technology_radar_vol_30_en.pdf"
}

document_loader.add_files(files=files)


# In[6]:


# Show the documents in the collection
# collections_documents = document_loader.load()


# ## Usage
# ### Use within a chain
# Below is a complete example of setting up a RAG pipeline with Needle within a chain:

# In[7]:


import os

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers.needle import NeedleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

# Initialize the Needle retriever (make sure your Needle API key is set as an environment variable)
retriever = NeedleRetriever(
    needle_api_key=os.getenv("NEEDLE_API_KEY"),
    collection_id="clt_01J87M9T6B71DHZTHNXYZQRG5H",
)

# Define system prompt for the assistant
system_prompt = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question.
    If you don't know, say so concisely.\n\n{context}
    """

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)

# Define the question-answering chain using a document chain (stuff chain) and the retriever
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Create the RAG (Retrieval-Augmented Generation) chain by combining the retriever and the question-answering chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Define the input query
query = {"input": "Did RAG move to accepted?"}

response = rag_chain.invoke(query)

response


# ## API reference
# 
# For detailed documentation of all `Needle` features and configurations head to the API reference: https://docs.needle-ai.com
