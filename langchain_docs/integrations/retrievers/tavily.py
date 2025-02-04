#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: TavilySearchAPI
---
# # TavilySearchAPIRetriever
# 
# >[Tavily's Search API](https://tavily.com) is a search engine built specifically for AI agents (LLMs), delivering real-time, accurate, and factual results at speed.
# 
# We can use this as a [retriever](/docs/how_to#retrievers). It will show functionality specific to this integration. After going through, it may be useful to explore [relevant use-case pages](/docs/how_to#qa-with-rag) to learn how to use this vectorstore as part of a larger chain.
# 
# ### Integration details
# 
# import {ItemTable} from "@theme/FeatureTables";
# 
# <ItemTable category="external_retrievers" item="TavilySearchAPIRetriever" />
# 
# ## Setup

# If you want to get automated tracing from individual queries, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[ ]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# The integration lives in the `langchain-community` package. We also need to install the `tavily-python` package itself.

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-community tavily-python')


# We also need to set our Tavily API key.

# In[ ]:


import getpass
import os

os.environ["TAVILY_API_KEY"] = getpass.getpass()


# ## Instantiation
# 
# Now we can instantiate our retriever:

# In[1]:


from langchain_community.retrievers import TavilySearchAPIRetriever

retriever = TavilySearchAPIRetriever(k=3)


# ## Usage

# In[2]:


query = "what year was breath of the wild released?"

retriever.invoke(query)


# ## Use within a chain
# 
# We can easily combine this retriever in to a chain.

# In[3]:


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)

llm = ChatOpenAI(model="gpt-4o-mini")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# In[4]:


chain.invoke("how many units did bretch of the wild sell in 2020")


# ## API reference
# 
# For detailed documentation of all `TavilySearchAPIRetriever` features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/retrievers/langchain_community.retrievers.tavily_search_api.TavilySearchAPIRetriever.html).
