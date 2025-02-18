#!/usr/bin/env python
# coding: utf-8

# # Motörhead
#
# >[Motörhead](https://github.com/getmetal/motorhead) is a memory server implemented in Rust. It automatically handles incremental summarization in the background and allows for stateless applications.
#
# ## Setup
#
# See instructions at [Motörhead](https://github.com/getmetal/motorhead) for running the server locally.

# In[ ]:


from langchain_community.memory.motorhead_memory import MotorheadMemory


# ## Example

# In[1]:


from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

template = """You are a chatbot having a conversation with a human.

{chat_history}
Human: {human_input}
AI:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)
memory = MotorheadMemory(
    session_id="testing-1", url="http://localhost:8080", memory_key="chat_history"
)

await memory.init()
# loads previous state from Motörhead 🤘

llm_chain = LLMChain(
    llm=OpenAI(),
    prompt=prompt,
    verbose=True,
    memory=memory,
)


# In[2]:


llm_chain.run("hi im bob")


# In[3]:


llm_chain.run("whats my name?")


# In[4]:


llm_chain.run("whats for dinner?")


# In[ ]:
