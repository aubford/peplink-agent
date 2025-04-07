#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('pip', 'install -qU langchain-airbyte langchain_chroma')


# In[3]:


import getpass

GITHUB_TOKEN = getpass.getpass()


# In[12]:


from langchain_airbyte import AirbyteLoader
from langchain_core.prompts import PromptTemplate

loader = AirbyteLoader(
    source="source-github",
    stream="pull_requests",
    config={
        "credentials": {"personal_access_token": GITHUB_TOKEN},
        "repositories": ["langchain-ai/langchain"],
    },
    template=PromptTemplate.from_template(
        """# {title}
by {user[login]}

{body}"""
    ),
    include_metadata=False,
)
docs = loader.load()


# In[19]:


print(docs[-2].page_content)


# In[39]:


len(docs)


# In[29]:


import tiktoken
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

enc = tiktoken.get_encoding("cl100k_base")

vectorstore = Chroma.from_documents(
    docs,
    embedding=OpenAIEmbeddings(
        disallowed_special=(enc.special_tokens_set - {"<|endofprompt|>"})
    ),
)


# In[40]:


retriever = vectorstore.as_retriever()


# In[42]:


retriever.invoke("pull requests related to IBM")


# In[ ]:




