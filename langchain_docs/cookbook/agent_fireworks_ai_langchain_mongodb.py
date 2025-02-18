#!/usr/bin/env python
# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/agents/agent_fireworks_ai_langchain_mongodb.ipynb)
#
# [![View Article](https://img.shields.io/badge/View%20Article-blue)](https://www.mongodb.com/developer/products/atlas/agent-fireworksai-mongodb-langchain/)

# ## Install Libraries

# In[ ]:


get_ipython().system(
    "pip install langchain langchain_openai langchain-fireworks langchain-mongodb arxiv pymupdf datasets pymongo"
)


# ## Set Evironment Variables

# In[3]:


import os

os.environ["OPENAI_API_KEY"] = ""
os.environ["FIREWORKS_API_KEY"] = ""
os.environ["MONGO_URI"] = ""

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")


# ## Data Ingestion into MongoDB Vector Database
#

# In[2]:


import pandas as pd
from datasets import load_dataset

data = load_dataset("MongoDB/subset_arxiv_papers_with_emebeddings")
dataset_df = pd.DataFrame(data["train"])


# In[4]:


print(len(dataset_df))
dataset_df.head()


# In[5]:


from pymongo import MongoClient

# Initialize MongoDB python client
client = MongoClient(MONGO_URI, appname="devrel.content.ai_agent_firechain.python")

DB_NAME = "agent_demo"
COLLECTION_NAME = "knowledge"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
collection = client[DB_NAME][COLLECTION_NAME]


# In[6]:


# Delete any existing records in the collection
collection.delete_many({})

# Data Ingestion
records = dataset_df.to_dict("records")
collection.insert_many(records)

print("Data ingestion into MongoDB completed")


# ## Create Vector Search Index Defintion
#
# ```
# {
#   "fields": [
#     {
#       "type": "vector",
#       "path": "embedding",
#       "numDimensions": 256,
#       "similarity": "cosine"
#     }
#   ]
# }
# ```

# ## Create LangChain Retriever (MongoDB)

# In[7]:


from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=256)

# Vector Store Creation
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGO_URI,
    namespace=DB_NAME + "." + COLLECTION_NAME,
    embedding=embedding_model,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    text_key="abstract",
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})


# ### Optional: Creating a retrevier with compression capabilities using LLMLingua
#

# In[ ]:


get_ipython().system("pip install langchain_community llmlingua")


# In[9]:


from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import LLMLinguaCompressor


# In[12]:


compressor = LLMLinguaCompressor(model_name="openai-community/gpt2", device_map="cpu")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


# ## Configure LLM Using Fireworks AI

# In[61]:


from langchain_fireworks import ChatFireworks

llm = ChatFireworks(model="accounts/fireworks/models/firefunction-v1", max_tokens=256)


# ## Agent Tools Creation

# In[51]:


from langchain.agents import tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import ArxivLoader


# Custom Tool Definiton
@tool
def get_metadata_information_from_arxiv(word: str) -> list:
    """
    Fetches and returns metadata for a maximum of ten documents from arXiv matching the given query word.

    Args:
      word (str): The search query to find relevant documents on arXiv.

    Returns:
      list: Metadata about the documents matching the query.
    """
    docs = ArxivLoader(query=word, load_max_docs=10).load()
    # Extract just the metadata from each document
    metadata_list = [doc.metadata for doc in docs]
    return metadata_list


@tool
def get_information_from_arxiv(word: str) -> list:
    """
    Fetches and returns metadata for a single research paper from arXiv matching the given query word, which is the ID of the paper, for example: 704.0001.

    Args:
      word (str): The search query to find the relevant paper on arXiv using the ID.

    Returns:
      list: Data about the paper matching the query.
    """
    doc = ArxivLoader(query=word, load_max_docs=1).load()
    return doc


# If you created a retriever with compression capaitilies in the optional cell in an earlier cell, you can replace 'retriever' with 'compression_retriever'
# Otherwise you can also create a compression procedure as a tool for the agent as shown in the `compress_prompt_using_llmlingua` tool definition function
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="knowledge_base",
    description="This serves as the base knowledge source of the agent and contains some records of research papers from Arxiv. This tool is used as the first step for exploration and reseach efforts.",
)


# In[52]:


from langchain_community.document_compressors import LLMLinguaCompressor

compressor = LLMLinguaCompressor(model_name="openai-community/gpt2", device_map="cpu")


@tool
def compress_prompt_using_llmlingua(prompt: str, compression_rate: float = 0.5) -> str:
    """
    Compresses a long data or prompt using the LLMLinguaCompressor.

    Args:
        data (str): The data or prompt to be compressed.
        compression_rate (float): The rate at which to compress the data (default is 0.5).

    Returns:
        str: The compressed data or prompt.
    """
    compressed_data = compressor.compress_prompt(
        prompt,
        rate=compression_rate,
        force_tokens=["!", ".", "?", "\n"],
        drop_consecutive=True,
    )
    return compressed_data


# In[53]:


tools = [
    retriever_tool,
    get_metadata_information_from_arxiv,
    get_information_from_arxiv,
    compress_prompt_using_llmlingua,
]


# ## Agent Prompt Creation

# In[89]:


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

agent_purpose = """
You are a helpful research assistant equipped with various tools to assist with your tasks efficiently. 
You have access to conversational history stored in your inpout as chat_history.
You are cost-effective and utilize the compress_prompt_using_llmlingua tool whenever you determine that a prompt or conversational history is too long. 
Below are instructions on when and how to use each tool in your operations.

1. get_metadata_information_from_arxiv

Purpose: To fetch and return metadata for up to ten documents from arXiv that match a given query word.
When to Use: Use this tool when you need to gather metadata about multiple research papers related to a specific topic.
Example: If you are asked to provide an overview of recent papers on "machine learning," use this tool to fetch metadata for relevant documents.

2. get_information_from_arxiv

Purpose: To fetch and return metadata for a single research paper from arXiv using the paper's ID.
When to Use: Use this tool when you need detailed information about a specific research paper identified by its arXiv ID.
Example: If you are asked to retrieve detailed information about the paper with the ID "704.0001," use this tool.

3. retriever_tool

Purpose: To serve as your base knowledge, containing records of research papers from arXiv.
When to Use: Use this tool as the first step for exploration and research efforts when dealing with topics covered by the documents in the knowledge base.
Example: When beginning research on a new topic that is well-documented in the arXiv repository, use this tool to access the relevant papers.

4. compress_prompt_using_llmlingua

Purpose: To compress long prompts or conversational histories using the LLMLinguaCompressor.
When to Use: Use this tool whenever you determine that a prompt or conversational history is too long to be efficiently processed.
Example: If you receive a very lengthy query or conversation context that exceeds the typical token limits, compress it using this tool before proceeding with further processing.

"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", agent_purpose),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)


# ## Agent Memory Using MongoDB

# In[92]:


from langchain.memory import ConversationBufferMemory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory


def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        MONGO_URI, session_id, database_name=DB_NAME, collection_name="history"
    )


memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=get_session_history("latest_agent_session")
)


# ## Agent Creation

# In[93]:


from langchain.agents import AgentExecutor, create_tool_calling_agent

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
)


# ## Agent Exectution

# In[94]:


agent_executor.invoke(
    {
        "input": "Get me a list of research papers on the topic Prompt Compression in LLM Applications."
    }
)


# In[95]:


agent_executor.invoke({"input": "What paper did we speak about from our chat history?"})
