#!/usr/bin/env python
# coding: utf-8
---
sidebar_position: 3
---
# # How to handle cases where no queries are generated
# 
# Sometimes, a query analysis technique may allow for any number of queries to be generated - including no queries! In this case, our overall chain will need to inspect the result of the query analysis before deciding whether to call the retriever or not.
# 
# We will use mock data for this example.

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
vectorstore = Chroma.from_texts(
    texts,
    embeddings,
)
retriever = vectorstore.as_retriever()


# ## Query analysis
# 
# We will use function calling to structure the output. However, we will configure the LLM such that is doesn't NEED to call the function representing a search query (should it decide not to). We will also then use a prompt to do query analysis that explicitly lays when it should and shouldn't make a search.

# In[4]:


from typing import Optional

from pydantic import BaseModel, Field


class Search(BaseModel):
    """Search over a database of job records."""

    query: str = Field(
        ...,
        description="Similarity search query applied to job record.",
    )


# In[5]:


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

system = """You have the ability to issue search queries to get information to help answer user information.

You do not NEED to look things up. If you don't need to, then just respond normally."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.bind_tools([Search])
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm


# We can see that by invoking this we get an message that sometimes - but not always - returns a tool call.

# In[6]:


query_analyzer.invoke("where did Harrison Work")


# In[7]:


query_analyzer.invoke("hi!")


# ## Retrieval with query analysis
# 
# So how would we include this in a chain? Let's look at an example below.

# In[8]:


from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.runnables import chain

output_parser = PydanticToolsParser(tools=[Search])


# In[9]:


@chain
def custom_chain(question):
    response = query_analyzer.invoke(question)
    if "tool_calls" in response.additional_kwargs:
        query = output_parser.invoke(response)
        docs = retriever.invoke(query[0].query)
        # Could add more logic - like another LLM call - here
        return docs
    else:
        return response


# In[10]:


custom_chain.invoke("where did Harrison Work")


# In[11]:


custom_chain.invoke("hi!")


# In[ ]:




