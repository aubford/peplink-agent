#!/usr/bin/env python
# coding: utf-8

# # Cohere RAG
# 
# >[Cohere](https://cohere.ai/about) is a Canadian startup that provides natural language processing models that help companies improve human-machine interactions.
# 
# This notebook covers how to get started with the `Cohere RAG` retriever. This allows you to leverage the ability to search documents over various connectors or by supplying your own.

# In[ ]:


import getpass
import os

os.environ["COHERE_API_KEY"] = getpass.getpass()


# In[ ]:


from langchain_cohere import ChatCohere, CohereRagRetriever
from langchain_core.documents import Document


# In[ ]:


rag = CohereRagRetriever(llm=ChatCohere())


# In[3]:


def _pretty_print(docs):
    for doc in docs:
        print(doc.metadata)
        print("\n\n" + doc.page_content)
        print("\n\n" + "-" * 30 + "\n\n")


# In[4]:


_pretty_print(rag.invoke("What is cohere ai?"))


# In[5]:


_pretty_print(await rag.ainvoke("What is cohere ai?"))  # async version


# In[6]:


docs = rag.invoke(
    "Does langchain support cohere RAG?",
    documents=[
        Document(page_content="Langchain supports cohere RAG!"),
        Document(page_content="The sky is blue!"),
    ],
)
_pretty_print(docs)


# Please note that connectors and documents cannot be used simultaneously. If you choose to provide documents in the `invoke` method, they will take precedence, and connectors will not be utilized for that particular request, as shown in the snippet above!

# In[ ]:




