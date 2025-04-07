#!/usr/bin/env python
# coding: utf-8

# # ZepCloudChatMessageHistory
# > Recall, understand, and extract data from chat histories. Power personalized AI experiences.
# 
# >[Zep](https://www.getzep.com) is a long-term memory service for AI Assistant apps.
# > With Zep, you can provide AI assistants with the ability to recall past conversations, no matter how distant,
# > while also reducing hallucinations, latency, and cost.
# 
# > See [Zep Cloud Installation Guide](https://help.getzep.com/sdks) and more [Zep Cloud Langchain Examples](https://github.com/getzep/zep-python/tree/main/examples)
# 
# ## Example
# 
# This notebook demonstrates how to use [Zep](https://www.getzep.com/) to persist chat history and use Zep Memory with your chain.
# 

# In[7]:


from uuid import uuid4

from langchain_community.chat_message_histories import ZepCloudChatMessageHistory
from langchain_community.memory.zep_cloud_memory import ZepCloudMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableParallel,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

session_id = str(uuid4())  # This is a unique identifier for the session


# Provide your OpenAI key

# In[8]:


import getpass

openai_key = getpass.getpass()


# Provide your Zep API key. See https://help.getzep.com/projects#api-keys
# 

# In[9]:


zep_api_key = getpass.getpass()


# Preload some messages into the memory. The default message window is 4 messages. We want to push beyond this to demonstrate auto-summarization.

# In[10]:


test_history = [
    {"role": "human", "content": "Who was Octavia Butler?"},
    {
        "role": "ai",
        "content": (
            "Octavia Estelle Butler (June 22, 1947 – February 24, 2006) was an American"
            " science fiction author."
        ),
    },
    {"role": "human", "content": "Which books of hers were made into movies?"},
    {
        "role": "ai",
        "content": (
            "The most well-known adaptation of Octavia Butler's work is the FX series"
            " Kindred, based on her novel of the same name."
        ),
    },
    {"role": "human", "content": "Who were her contemporaries?"},
    {
        "role": "ai",
        "content": (
            "Octavia Butler's contemporaries included Ursula K. Le Guin, Samuel R."
            " Delany, and Joanna Russ."
        ),
    },
    {"role": "human", "content": "What awards did she win?"},
    {
        "role": "ai",
        "content": (
            "Octavia Butler won the Hugo Award, the Nebula Award, and the MacArthur"
            " Fellowship."
        ),
    },
    {
        "role": "human",
        "content": "Which other women sci-fi writers might I want to read?",
    },
    {
        "role": "ai",
        "content": "You might want to read Ursula K. Le Guin or Joanna Russ.",
    },
    {
        "role": "human",
        "content": (
            "Write a short synopsis of Butler's book, Parable of the Sower. What is it"
            " about?"
        ),
    },
    {
        "role": "ai",
        "content": (
            "Parable of the Sower is a science fiction novel by Octavia Butler,"
            " published in 1993. It follows the story of Lauren Olamina, a young woman"
            " living in a dystopian future where society has collapsed due to"
            " environmental disasters, poverty, and violence."
        ),
        "metadata": {"foo": "bar"},
    },
]

zep_memory = ZepCloudMemory(
    session_id=session_id,
    api_key=zep_api_key,
)

for msg in test_history:
    zep_memory.chat_memory.add_message(
        HumanMessage(content=msg["content"])
        if msg["role"] == "human"
        else AIMessage(content=msg["content"])
    )

import time

time.sleep(
    10
)  # Wait for the messages to be embedded and summarized, this happens asynchronously.


# **MessagesPlaceholder** - We’re using the variable name chat_history here. This will incorporate the chat history into the prompt.
# It’s important that this variable name aligns with the history_messages_key in the RunnableWithMessageHistory chain for seamless integration.
# 
# **question** must match input_messages_key in `RunnableWithMessageHistory“ chain.

# In[11]:


template = """Be helpful and answer the question below using the provided context:
    """
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)


# We use RunnableWithMessageHistory to incorporate Zep’s Chat History into our chain. This class requires a session_id as a parameter when you activate the chain.

# In[12]:


inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"],
    },
)
chain = RunnableWithMessageHistory(
    inputs | answer_prompt | ChatOpenAI(openai_api_key=openai_key) | StrOutputParser(),
    lambda s_id: ZepCloudChatMessageHistory(
        session_id=s_id,  # This uniquely identifies the conversation, note that we are getting session id as chain configurable field
        api_key=zep_api_key,
        memory_type="perpetual",
    ),
    input_messages_key="question",
    history_messages_key="chat_history",
)


# In[13]:


chain.invoke(
    {
        "question": "What is the book's relevance to the challenges facing contemporary society?"
    },
    config={"configurable": {"session_id": session_id}},
)


# In[ ]:




