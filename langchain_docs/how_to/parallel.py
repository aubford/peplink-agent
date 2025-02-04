#!/usr/bin/env python
# coding: utf-8
---
sidebar_position: 1
keywords: [RunnableParallel, RunnableMap, LCEL]
---
# # How to invoke runnables in parallel
# 
# :::info Prerequisites
# 
# This guide assumes familiarity with the following concepts:
# - [LangChain Expression Language (LCEL)](/docs/concepts/lcel)
# - [Chaining runnables](/docs/how_to/sequence)
# 
# :::
# 
# The [`RunnableParallel`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableParallel.html) primitive is essentially a dict whose values are runnables (or things that can be coerced to runnables, like functions). It runs all of its values in parallel, and each value is called with the overall input of the `RunnableParallel`. The final return value is a dict with the results of each value under its appropriate key.
# 
# ## Formatting with `RunnableParallels`
# 
# `RunnableParallels` are useful for parallelizing operations, but can also be useful for manipulating the output of one Runnable to match the input format of the next Runnable in a sequence. You can use them to split or fork the chain so that multiple components can process the input in parallel. Later, other components can join or merge the results to synthesize a final response. This type of chain creates a computation graph that looks like the following:
# 
# ```text
#      Input
#       / \
#      /   \
#  Branch1 Branch2
#      \   /
#       \ /
#       Combine
# ```
# 
# Below, the input to prompt is expected to be a map with keys `"context"` and `"question"`. The user input is just the question. So we need to get the context using our retriever and passthrough the user input under the `"question"` key.
# 

# In[ ]:


# | output: false
# | echo: false

get_ipython().run_line_magic('pip', 'install -qU langchain langchain_openai')

import os
from getpass import getpass

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass()


# In[2]:


from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

# The prompt expects input with keys for "context" and "question"
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

retrieval_chain.invoke("where did harrison work?")


# :::tip
# Note that when composing a RunnableParallel with another Runnable we don't even need to wrap our dictionary in the RunnableParallel class — the type conversion is handled for us. In the context of a chain, these are equivalent:
# :::
# 
# ```
# {"context": retriever, "question": RunnablePassthrough()}
# ```
# 
# ```
# RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
# ```
# 
# ```
# RunnableParallel(context=retriever, question=RunnablePassthrough())
# ```
# 
# See the section on [coercion for more](/docs/how_to/sequence/#coercion).

# ## Using itemgetter as shorthand
# 
# Note that you can use Python's `itemgetter` as shorthand to extract data from the map when combining with `RunnableParallel`. You can find more information about itemgetter in the [Python Documentation](https://docs.python.org/3/library/operator.html#operator.itemgetter). 
# 
# In the example below, we use itemgetter to extract specific keys from the map:

# In[3]:


from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke({"question": "where did harrison work", "language": "italian"})


# ## Parallelize steps
# 
# RunnableParallels make it easy to execute multiple Runnables in parallel, and to return the output of these Runnables as a map.

# In[4]:


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

model = ChatOpenAI()
joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
poem_chain = (
    ChatPromptTemplate.from_template("write a 2-line poem about {topic}") | model
)

map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)

map_chain.invoke({"topic": "bear"})


# ## Parallelism
# 
# RunnableParallel are also useful for running independent processes in parallel, since each Runnable in the map is executed in parallel. For example, we can see our earlier `joke_chain`, `poem_chain` and `map_chain` all have about the same runtime, even though `map_chain` executes both of the other two.

# In[5]:


get_ipython().run_cell_magic('timeit', '', '\njoke_chain.invoke({"topic": "bear"})\n')


# In[6]:


get_ipython().run_cell_magic('timeit', '', '\npoem_chain.invoke({"topic": "bear"})\n')


# In[7]:


get_ipython().run_cell_magic('timeit', '', '\nmap_chain.invoke({"topic": "bear"})\n')


# ## Next steps
# 
# You now know some ways to format and parallelize chain steps with `RunnableParallel`.
# 
# To learn more, see the other how-to guides on runnables in this section.

# In[ ]:




