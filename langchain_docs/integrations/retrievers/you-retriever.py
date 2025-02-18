#!/usr/bin/env python
# coding: utf-8

# # You.com
#
# >[you.com API](https://api.you.com) is a suite of tools designed to help developers ground the output of LLMs in the most recent, most accurate, most relevant information that may not have been included in their training dataset.

# ## Setup

# The retriever lives in the `langchain-community` package.
#
# You also need to set your you.com API key.

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet langchain-community")


# In[4]:


import os

os.environ["YDC_API_KEY"] = ""

# For use in Chaining section
os.environ["OPENAI_API_KEY"] = ""

## ALTERNATIVE: load YDC_API_KEY from a .env file

# !pip install --quiet -U python-dotenv
# import dotenv
# dotenv.load_dotenv()


# It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability

# In[5]:


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
# os.environ["LANGSMITH_PROJECT"] = 'Experimentz'


# ## Utility Usage

# In[ ]:


from langchain_community.utilities import YouSearchAPIWrapper

utility = YouSearchAPIWrapper(num_web_results=1)

utility


# In[77]:


import json

# .raw_results returns the unaltered response from the API
response = utility.raw_results(query="What is the weather in NY")
# API returns an object with a `hits` key containing a list of hits
hits = response["hits"]

# with `num_web_results=1`, `hits` should be len of 1
print(len(hits))

print(json.dumps(hits, indent=2))


# In[78]:


# .results returns parsed results with each snippet in a Document
response = utility.results(query="What is the weather in NY")

# .results should have a Document for each `snippet`
print(len(response))

print(response)


# ## Retriever Usage

# In[ ]:


from langchain_community.retrievers.you import YouRetriever

retriever = YouRetriever(num_web_results=1)

retriever


# In[95]:


# .invoke wraps utility.results
response = retriever.invoke("What is the weather in NY")

# .invoke should have a Document for each `snippet`
print(len(response))

print(response)


# ## Chaining

# In[ ]:


# you need a model to use in the chain
get_ipython().system("pip install --upgrade --quiet langchain-openai")


# In[98]:


from langchain_community.retrievers.you import YouRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# set up runnable
runnable = RunnablePassthrough

# set up retriever, limit sources to one
retriever = YouRetriever(num_web_results=1)

# set up model
model = ChatOpenAI(model="gpt-3.5-turbo-16k")

# set up output parser
output_parser = StrOutputParser()


# ### Invoke

# In[99]:


# set up prompt that expects one question
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)

# set up chain
chain = (
    runnable.assign(context=(lambda x: x["question"]) | retriever)
    | prompt
    | model
    | output_parser
)

output = chain.invoke({"question": "what is the weather in NY today"})

print(output)


# ### Stream

# In[100]:


# set up prompt that expects one question
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)

# set up chain - same as above
chain = (
    runnable.assign(context=(lambda x: x["question"]) | retriever)
    | prompt
    | model
    | output_parser
)

for s in chain.stream({"question": "what is the weather in NY today"}):
    print(s, end="", flush=True)


# ### Batch

# In[101]:


chain = (
    runnable.assign(context=(lambda x: x["question"]) | retriever)
    | prompt
    | model
    | output_parser
)

output = chain.batch(
    [
        {"question": "what is the weather in NY today"},
        {"question": "what is the weather in sf today"},
    ]
)

for o in output:
    print(o)
