#!/usr/bin/env python
# coding: utf-8

# # Outline
#
# >[Outline](https://www.getoutline.com/) is an open-source collaborative knowledge base platform designed for team information sharing.
#
# This notebook shows how to retrieve documents from your Outline instance into the Document format that is used downstream.

# ## Setup

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet langchain langchain-openai"
)


# You first need to [create an api key](https://www.getoutline.com/developers#section/Authentication) for your Outline instance. Then you need to set the following environment variables:

# In[1]:


import os

os.environ["OUTLINE_API_KEY"] = "xxx"
os.environ["OUTLINE_INSTANCE_URL"] = "https://app.getoutline.com"


# `OutlineRetriever` has these arguments:
# - optional `top_k_results`: default=3. Use it to limit number of documents retrieved.
# - optional `load_all_available_meta`: default=False. By default only the most important fields retrieved: `title`, `source` (the url of the document). If True, other fields also retrieved.
# - optional `doc_content_chars_max` default=4000. Use it to limit the number of characters for each document retrieved.
#
# `get_relevant_documents()` has one argument, `query`: free text which used to find documents in your Outline instance.

# ## Examples

# ### Running retriever

# In[2]:


from langchain_community.retrievers import OutlineRetriever


# In[3]:


retriever = OutlineRetriever()


# In[4]:


retriever.invoke("LangChain", doc_content_chars_max=100)


# ### Answering Questions on Outline Documents

# In[5]:


import os
from getpass import getpass

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("OpenAI API Key:")


# In[6]:


from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)


# In[7]:


qa({"question": "what is langchain?", "chat_history": {}})
