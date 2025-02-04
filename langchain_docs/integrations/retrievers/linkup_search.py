#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: LinkupSearchRetriever
---
# # LinkupSearchRetriever
# 
# > [Linkup](https://www.linkup.so/) provides an API to connect LLMs to the web and the Linkup Premium Partner sources.
# 
# This will help you getting started with the LinkupSearchRetriever [retriever](/docs/concepts/retrievers/). For detailed documentation of all LinkupSearchRetriever features and configurations head to the [API reference](https://python.langchain.com/api_reference/linkup/retrievers/linkup_langchain.search_retriever.LinkupSearchRetriever.html).
# 
# ### Integration details
# 
# | Retriever | Source | Package |
# | :--- | :--- | :---: |
# [LinkupSearchRetriever](https://python.langchain.com/api_reference/linkup/retrievers/linkup_langchain.search_retriever.LinkupSearchRetriever.html) | Web and partner sources | langchain-linkup |
# 
# ## Setup
# 
# To use the Linkup provider, you need a valid API key, which you can find by signing-up [here](https://app.linkup.so/sign-up). You can then set it up as the `LINKUP_API_KEY` environment variable. For the chain example below, you also need to set an OpenAI API key as `OPENAI_API_KEY` environment variable, which you can also do here:

# In[ ]:


# import os
# os.environ["LINKUP_API_KEY"] = ""  # Fill with your API key
# os.environ["OPENAI_API_KEY"] = ""  # Fill with your API key


# If you want to get automated tracing from individual queries, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[ ]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# This retriever lives in the `langchain-linkup` package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-linkup')


# ## Instantiation
# 
# Now we can instantiate our retriever:

# In[ ]:


from langchain_linkup import LinkupSearchRetriever

retriever = LinkupSearchRetriever(
    depth="deep",  # "standard" or "deep"
    linkup_api_key=None,  # API key can be passed here or set as the LINKUP_API_KEY environment variable
)


# ## Usage

# In[6]:


query = "Who won the latest US presidential elections?"

retriever.invoke(query)


# ## Use within a chain
# 
# Like other retrievers, LinkupSearchRetriever can be incorporated into LLM applications via [chains](/docs/how_to/sequence/).
# 
# We will need a LLM or chat model:
# 
# ```{=mdx}
# import ChatModelTabs from "@theme/ChatModelTabs";
# 
# <ChatModelTabs customVarName="llm" />
# ```

# In[ ]:


# | output: false
# | echo: false

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)


# In[ ]:


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# In[9]:


chain.invoke("Who won the 3 latest US presidential elections?")


# ## API reference
# 
# For detailed documentation of all LinkupSearchRetriever features and configurations head to the [API reference](https://python.langchain.com/api_reference/linkup/retrievers/linkup_langchain.search_retriever.LinkupSearchRetriever.html).
