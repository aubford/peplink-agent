#!/usr/bin/env python
# coding: utf-8

# # Combine agents and vector stores
# 
# This notebook covers how to combine agents and vector stores. The use case for this is that you've ingested your data into a vector store and want to interact with it in an agentic manner.
# 
# The recommended method for doing so is to create a `RetrievalQA` and then use that as a tool in the overall agent. Let's take a look at doing this below. You can do this with multiple different vector DBs, and use the agent as a way to route between them. There are two different ways of doing this - you can either let the agent use the vector stores as normal tools, or you can set `return_direct=True` to really just use the agent as a router.

# ## Create the vector store

# In[16]:


from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

llm = OpenAI(temperature=0)


# In[17]:


from pathlib import Path

relevant_parts = []
for p in Path(".").absolute().parts:
    relevant_parts.append(p)
    if relevant_parts[-3:] == ["langchain", "docs", "modules"]:
        break
doc_path = str(Path(*relevant_parts) / "state_of_the_union.txt")


# In[18]:


from langchain_community.document_loaders import TextLoader

loader = TextLoader(doc_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings, collection_name="state-of-union")


# In[4]:


state_of_union = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
)


# In[5]:


from langchain_community.document_loaders import WebBaseLoader


# In[6]:


loader = WebBaseLoader("https://beta.ruff.rs/docs/faq/")


# In[7]:


docs = loader.load()
ruff_texts = text_splitter.split_documents(docs)
ruff_db = Chroma.from_documents(ruff_texts, embeddings, collection_name="ruff")
ruff = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=ruff_db.as_retriever()
)


# In[ ]:





# ## Create the Agent

# In[43]:


# Import things that are needed generically
from langchain.agents import AgentType, Tool, initialize_agent
from langchain_openai import OpenAI


# In[44]:


tools = [
    Tool(
        name="State of Union QA System",
        func=state_of_union.run,
        description="useful for when you need to answer questions about the most recent state of the union address. Input should be a fully formed question.",
    ),
    Tool(
        name="Ruff QA System",
        func=ruff.run,
        description="useful for when you need to answer questions about ruff (a python linter). Input should be a fully formed question.",
    ),
]


# In[45]:


# Construct the agent. We will use the default agent type here.
# See documentation for a full list of options.
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


# In[46]:


agent.run(
    "What did biden say about ketanji brown jackson in the state of the union address?"
)


# In[47]:


agent.run("Why use ruff over flake8?")


# ## Use the Agent solely as a router

# You can also set `return_direct=True` if you intend to use the agent as a router and just want to directly return the result of the RetrievalQAChain.
# 
# Notice that in the above examples the agent did some extra work after querying the RetrievalQAChain. You can avoid that and just return the result directly.

# In[48]:


tools = [
    Tool(
        name="State of Union QA System",
        func=state_of_union.run,
        description="useful for when you need to answer questions about the most recent state of the union address. Input should be a fully formed question.",
        return_direct=True,
    ),
    Tool(
        name="Ruff QA System",
        func=ruff.run,
        description="useful for when you need to answer questions about ruff (a python linter). Input should be a fully formed question.",
        return_direct=True,
    ),
]


# In[49]:


agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


# In[50]:


agent.run(
    "What did biden say about ketanji brown jackson in the state of the union address?"
)


# In[51]:


agent.run("Why use ruff over flake8?")


# ## Multi-Hop vector store reasoning
# 
# Because vector stores are easily usable as tools in agents, it is easy to use answer multi-hop questions that depend on vector stores using the existing agent framework.

# In[57]:


tools = [
    Tool(
        name="State of Union QA System",
        func=state_of_union.run,
        description="useful for when you need to answer questions about the most recent state of the union address. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before.",
    ),
    Tool(
        name="Ruff QA System",
        func=ruff.run,
        description="useful for when you need to answer questions about ruff (a python linter). Input should be a fully formed question, not referencing any obscure pronouns from the conversation before.",
    ),
]


# In[58]:


# Construct the agent. We will use the default agent type here.
# See documentation for a full list of options.
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


# In[59]:


agent.run(
    "What tool does ruff use to run over Jupyter Notebooks? Did the president mention that tool in the state of the union?"
)


# In[ ]:




