#!/usr/bin/env python
# coding: utf-8

# # KDB.AI
# 
# > [KDB.AI](https://kdb.ai/) is a powerful knowledge-based vector database and search engine that allows you to build scalable, reliable AI applications, using real-time data, by providing advanced search, recommendation and personalization.
# 
# [This example](https://github.com/KxSystems/kdbai-samples/blob/main/document_search/document_search.ipynb) demonstrates how to use KDB.AI to run semantic search on unstructured text documents.
# 
# To access your end point and API keys, [sign up to KDB.AI here](https://kdb.ai/get-started/).
# 
# To set up your development environment, follow the instructions on the [KDB.AI pre-requisites page](https://code.kx.com/kdbai/pre-requisites.html).
# 
# The following examples demonstrate some of the ways you can interact with KDB.AI through LangChain.
# 
# You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration
# 
# ## Import required packages

# In[1]:


import os
import time
from getpass import getpass

import kdbai_client as kdbai
import pandas as pd
import requests
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import KDBAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# In[2]:


KDBAI_ENDPOINT = input("KDB.AI endpoint: ")
KDBAI_API_KEY = getpass("KDB.AI API key: ")
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("OpenAI API Key: ")


# In[3]:


TEMP = 0.0
K = 3


# ## Create a KBD.AI Session

# In[4]:


print("Create a KDB.AI session...")
session = kdbai.Session(endpoint=KDBAI_ENDPOINT, api_key=KDBAI_API_KEY)


# ## Create a table

# In[5]:


print('Create table "documents"...')
schema = {
    "columns": [
        {"name": "id", "pytype": "str"},
        {"name": "text", "pytype": "bytes"},
        {
            "name": "embeddings",
            "pytype": "float32",
            "vectorIndex": {"dims": 1536, "metric": "L2", "type": "hnsw"},
        },
        {"name": "tag", "pytype": "str"},
        {"name": "title", "pytype": "bytes"},
    ]
}
table = session.create_table("documents", schema)


# In[6]:


get_ipython().run_cell_magic('time', '', 'URL = "https://www.conseil-constitutionnel.fr/node/3850/pdf"\nPDF = "Déclaration_des_droits_de_l_homme_et_du_citoyen.pdf"\nopen(PDF, "wb").write(requests.get(URL).content)\n')


# ## Read a PDF

# In[7]:


get_ipython().run_cell_magic('time', '', 'print("Read a PDF...")\nloader = PyPDFLoader(PDF)\npages = loader.load_and_split()\nlen(pages)\n')


# ## Create a Vector Database from PDF Text

# In[8]:


get_ipython().run_cell_magic('time', '', 'print("Create a Vector Database from PDF text...")\nembeddings = OpenAIEmbeddings(model="text-embedding-ada-002")\ntexts = [p.page_content for p in pages]\nmetadata = pd.DataFrame(index=list(range(len(texts))))\nmetadata["tag"] = "law"\nmetadata["title"] = "Déclaration des Droits de l\'Homme et du Citoyen de 1789".encode(\n    "utf-8"\n)\nvectordb = KDBAI(table, embeddings)\nvectordb.add_texts(texts=texts, metadatas=metadata)\n')


# ## Create LangChain Pipeline

# In[9]:


get_ipython().run_cell_magic('time', '', 'print("Create LangChain Pipeline...")\nqabot = RetrievalQA.from_chain_type(\n    chain_type="stuff",\n    llm=ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=TEMP),\n    retriever=vectordb.as_retriever(search_kwargs=dict(k=K)),\n    return_source_documents=True,\n)\n')


# ## Summarize the document in English

# In[10]:


get_ipython().run_cell_magic('time', '', 'Q = "Summarize the document in English:"\nprint(f"\\n\\n{Q}\\n")\nprint(qabot.invoke(dict(query=Q))["result"])\n')


# ## Query the Data

# In[11]:


get_ipython().run_cell_magic('time', '', 'Q = "Is it a fair law and why ?"\nprint(f"\\n\\n{Q}\\n")\nprint(qabot.invoke(dict(query=Q))["result"])\n')


# In[12]:


get_ipython().run_cell_magic('time', '', 'Q = "What are the rights and duties of the man, the citizen and the society ?"\nprint(f"\\n\\n{Q}\\n")\nprint(qabot.invoke(dict(query=Q))["result"])\n')


# In[13]:


get_ipython().run_cell_magic('time', '', 'Q = "Is this law practical ?"\nprint(f"\\n\\n{Q}\\n")\nprint(qabot.invoke(dict(query=Q))["result"])\n')


# ## Clean up the Documents table

# In[14]:


# Clean up KDB.AI "documents" table and index for similarity search
# so this notebook could be played again and again
session.table("documents").drop()


# In[ ]:




