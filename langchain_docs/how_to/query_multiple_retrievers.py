#!/usr/bin/env python
# coding: utf-8
---
sidebar_position: 5
---
# # How to handle multiple retrievers when doing query analysis
# 
# Sometimes, a query analysis technique may allow for selection of which [retriever](/docs/concepts/retrievers/) to use. To use this, you will need to add some logic to select the retriever to do. We will show a simple example (using mock data) of how to do that.

# ## Setup
# #### Install dependencies

# In[1]:


get_ipython().run_line_magic('pip', 'install -qU langchain langchain-community langchain-openai langchain-chroma')


# #### Set environment variables
# 
# We'll use OpenAI in this example:

# In[2]:


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass()

# Optional, uncomment to trace runs with LangSmith. Sign up here: https://smith.langchain.com.
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()


# ### Create Index
# 
# We will create a vectorstore over fake information.

# In[3]:


from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

texts = ["Harrison worked at Kensho"]
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_texts(texts, embeddings, collection_name="harrison")
retriever_harrison = vectorstore.as_retriever(search_kwargs={"k": 1})

texts = ["Ankush worked at Facebook"]
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_texts(texts, embeddings, collection_name="ankush")
retriever_ankush = vectorstore.as_retriever(search_kwargs={"k": 1})


# ## Query analysis
# 
# We will use function calling to structure the output. We will let it return multiple queries.

# In[4]:


from typing import List, Optional

from pydantic import BaseModel, Field


class Search(BaseModel):
    """Search for information about a person."""

    query: str = Field(
        ...,
        description="Query to look up",
    )
    person: str = Field(
        ...,
        description="Person to look things up for. Should be `HARRISON` or `ANKUSH`.",
    )


# In[5]:


from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

output_parser = PydanticToolsParser(tools=[Search])

system = """You have the ability to issue search queries to get information to help answer user information."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(Search)
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm


# We can see that this allows for routing between retrievers

# In[6]:


query_analyzer.invoke("where did Harrison Work")


# In[7]:


query_analyzer.invoke("where did ankush Work")


# ## Retrieval with query analysis
# 
# So how would we include this in a chain? We just need some simple logic to select the retriever and pass in the search query

# In[8]:


from langchain_core.runnables import chain


# In[9]:


retrievers = {
    "HARRISON": retriever_harrison,
    "ANKUSH": retriever_ankush,
}


# In[10]:


@chain
def custom_chain(question):
    response = query_analyzer.invoke(question)
    retriever = retrievers[response.person]
    return retriever.invoke(response.query)


# In[11]:


custom_chain.invoke("where did Harrison Work")


# In[12]:


custom_chain.invoke("where did ankush Work")


# In[ ]:




