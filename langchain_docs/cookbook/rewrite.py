#!/usr/bin/env python
# coding: utf-8

# # Rewrite-Retrieve-Read
# 
# **Rewrite-Retrieve-Read** is a method proposed in the paper [Query Rewriting for Retrieval-Augmented Large Language Models](https://arxiv.org/pdf/2305.14283.pdf)
# 
# > Because the original query can not be always optimal to retrieve for the LLM, especially in the real world... we first prompt an LLM to rewrite the queries, then conduct retrieval-augmented reading
# 
# We show how you can easily do that with LangChain Expression Language

# ## Baseline
# 
# Baseline RAG (**Retrieve-and-read**) can be done like the following:

# In[1]:


from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


# In[2]:


template = """Answer the users question based only on the following context:

<context>
{context}
</context>

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(temperature=0)

search = DuckDuckGoSearchAPIWrapper()


def retriever(query):
    return search.run(query)


# In[3]:


chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


# In[4]:


simple_query = "what is langchain?"


# In[5]:


chain.invoke(simple_query)


# While this is fine for well formatted queries, it can break down for more complicated queries

# In[6]:


distracted_query = "man that sam bankman fried trial was crazy! what is langchain?"


# In[7]:


chain.invoke(distracted_query)


# This is because the retriever does a bad job with these "distracted" queries

# In[8]:


retriever(distracted_query)


# ## Rewrite-Retrieve-Read Implementation
# 
# The main part is a rewriter to rewrite the search query

# In[9]:


template = """Provide a better search query for \
web search engine to answer the given question, end \
the queries with ’**’. Question: \
{x} Answer:"""
rewrite_prompt = ChatPromptTemplate.from_template(template)


# In[10]:


from langchain import hub

rewrite_prompt = hub.pull("langchain-ai/rewrite")


# In[11]:


print(rewrite_prompt.template)


# In[12]:


# Parser to remove the `**`


def _parse(text):
    return text.strip('"').strip("**")


# In[13]:


rewriter = rewrite_prompt | ChatOpenAI(temperature=0) | StrOutputParser() | _parse


# In[14]:


rewriter.invoke({"x": distracted_query})


# In[15]:


rewrite_retrieve_read_chain = (
    {
        "context": {"x": RunnablePassthrough()} | rewriter | retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)


# In[16]:


rewrite_retrieve_read_chain.invoke(distracted_query)


# In[ ]:




