#!/usr/bin/env python
# coding: utf-8

# [![View Article](https://img.shields.io/badge/View%20Article-blue)](https://www.mongodb.com/developer/products/atlas/advanced-rag-langchain-mongodb/)
# 

# # Adding Semantic Caching and Memory to your RAG Application using MongoDB and LangChain
# 
# In this notebook, we will see how to use the new MongoDBCache and MongoDBChatMessageHistory in your RAG application.
# 

# ## Step 1: Install required libraries
# 
# - **datasets**: Python library to get access to datasets available on Hugging Face Hub
# 
# - **langchain**: Python toolkit for LangChain
# 
# - **langchain-mongodb**: Python package to use MongoDB as a vector store, semantic cache, chat history store etc. in LangChain
# 
# - **langchain-openai**: Python package to use OpenAI models with LangChain
# 
# - **pymongo**: Python toolkit for MongoDB
# 
# - **pandas**: Python library for data analysis, exploration, and manipulation

# In[1]:


get_ipython().system(' pip install -qU datasets langchain langchain-mongodb langchain-openai pymongo pandas')


# ## Step 2: Setup pre-requisites
# 
# * Set the MongoDB connection string. Follow the steps [here](https://www.mongodb.com/docs/manual/reference/connection-string/) to get the connection string from the Atlas UI.
# 
# * Set the OpenAI API key. Steps to obtain an API key as [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)

# In[2]:


import getpass


# In[3]:


MONGODB_URI = getpass.getpass("Enter your MongoDB connection string:")


# In[4]:


OPENAI_API_KEY = getpass.getpass("Enter your OpenAI API key:")


# In[5]:


# Optional-- If you want to enable Langsmith -- good for debugging
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()


# ## Step 3: Download the dataset
# 
# We will be using MongoDB's [embedded_movies](https://huggingface.co/datasets/MongoDB/embedded_movies) dataset

# In[6]:


import pandas as pd
from datasets import load_dataset


# In[ ]:


# Ensure you have an HF_TOKEN in your development environment:
# access tokens can be created or copied from the Hugging Face platform (https://huggingface.co/docs/hub/en/security-tokens)

# Load MongoDB's embedded_movies dataset from Hugging Face
# https://huggingface.co/datasets/MongoDB/airbnb_embeddings

data = load_dataset("MongoDB/embedded_movies")


# In[8]:


df = pd.DataFrame(data["train"])


# ## Step 4: Data analysis
# 
# Make sure length of the dataset is what we expect, drop Nones etc.

# In[10]:


# Previewing the contents of the data
df.head(1)


# In[11]:


# Only keep records where the fullplot field is not null
df = df[df["fullplot"].notna()]


# In[12]:


# Renaming the embedding field to "embedding" -- required by LangChain
df.rename(columns={"plot_embedding": "embedding"}, inplace=True)


# ## Step 5: Create a simple RAG chain using MongoDB as the vector store

# In[13]:


from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Initialize MongoDB python client
client = MongoClient(MONGODB_URI, appname="devrel.content.python")

DB_NAME = "langchain_chatbot"
COLLECTION_NAME = "data"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
collection = client[DB_NAME][COLLECTION_NAME]


# In[14]:


# Delete any existing records in the collection
collection.delete_many({})


# In[16]:


# Data Ingestion
records = df.to_dict("records")
collection.insert_many(records)

print("Data ingestion into MongoDB completed")


# In[18]:


from langchain_openai import OpenAIEmbeddings

# Using the text-embedding-ada-002 since that's what was used to create embeddings in the movies dataset
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002"
)


# In[19]:


# Vector Store Creation
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGODB_URI,
    namespace=DB_NAME + "." + COLLECTION_NAME,
    embedding=embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    text_key="fullplot",
)


# In[49]:


# Using the MongoDB vector store as a retriever in a RAG chain
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})


# In[25]:


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Generate context using the retriever, and pass the user question through
retrieve = {
    "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
    "question": RunnablePassthrough(),
}
template = """Answer the question based only on the following context: \
{context}

Question: {question}
"""
# Defining the chat prompt
prompt = ChatPromptTemplate.from_template(template)
# Defining the model to be used for chat completion
model = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
# Parse output as a string
parse_output = StrOutputParser()

# Naive RAG chain
naive_rag_chain = retrieve | prompt | model | parse_output


# In[26]:


naive_rag_chain.invoke("What is the best movie to watch when sad?")


# ## Step 6: Create a RAG chain with chat history

# In[27]:


from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory


# In[29]:


def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        MONGODB_URI, session_id, database_name=DB_NAME, collection_name="history"
    )


# In[50]:


# Given a follow-up question and history, create a standalone question
standalone_system_prompt = """
Given a chat history and a follow-up question, rephrase the follow-up question to be a standalone question. \
Do NOT answer the question, just reformulate it if needed, otherwise return it as is. \
Only return the final standalone question. \
"""
standalone_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", standalone_system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

question_chain = standalone_question_prompt | model | parse_output


# In[51]:


# Generate context by passing output of the question_chain i.e. the standalone question to the retriever
retriever_chain = RunnablePassthrough.assign(
    context=question_chain
    | retriever
    | (lambda docs: "\n\n".join([d.page_content for d in docs]))
)


# In[55]:


# Create a prompt that includes the context, history and the follow-up question
rag_system_prompt = """Answer the question based only on the following context: \
{context}
"""
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


# In[56]:


# RAG chain
rag_chain = retriever_chain | rag_prompt | model | parse_output


# In[57]:


# RAG chain with history
with_message_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)
with_message_history.invoke(
    {"question": "What is the best movie to watch when sad?"},
    {"configurable": {"session_id": "1"}},
)


# In[58]:


with_message_history.invoke(
    {
        "question": "Hmmm..I don't want to watch that one. Can you suggest something else?"
    },
    {"configurable": {"session_id": "1"}},
)


# In[59]:


with_message_history.invoke(
    {"question": "How about something more light?"},
    {"configurable": {"session_id": "1"}},
)


# ## Step 7: Get faster responses using Semantic Cache
# 
# **NOTE:** Semantic cache only caches the input to the LLM. When using it in retrieval chains, remember that documents retrieved can change between runs resulting in cache misses for semantically similar queries.

# In[61]:


from langchain_core.globals import set_llm_cache
from langchain_mongodb.cache import MongoDBAtlasSemanticCache

set_llm_cache(
    MongoDBAtlasSemanticCache(
        connection_string=MONGODB_URI,
        embedding=embeddings,
        collection_name="semantic_cache",
        database_name=DB_NAME,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        wait_until_ready=True,  # Optional, waits until the cache is ready to be used
    )
)


# In[62]:


get_ipython().run_cell_magic('time', '', 'naive_rag_chain.invoke("What is the best movie to watch when sad?")\n')


# In[63]:


get_ipython().run_cell_magic('time', '', 'naive_rag_chain.invoke("What is the best movie to watch when sad?")\n')


# In[64]:


get_ipython().run_cell_magic('time', '', 'naive_rag_chain.invoke("Which movie do I watch when sad?")\n')

