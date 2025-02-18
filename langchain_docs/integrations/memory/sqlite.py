#!/usr/bin/env python
# coding: utf-8

# # SQLite
#
# >[SQLite](https://en.wikipedia.org/wiki/SQLite) is a database engine written in the C programming language. It is not a standalone app; rather, it is a library that software developers embed in their apps. As such, it belongs to the family of embedded databases. It is the most widely deployed database engine, as it is used by several of the top web browsers, operating systems, mobile phones, and other embedded systems.
#
# In this walkthrough we'll create a simple conversation chain which uses `ConversationEntityMemory` backed by a `SqliteEntityStore`.

# In[ ]:


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()


# ## Usage
#
# To use the storage you need to provide only 2 things:
#
# 1. Session Id - a unique identifier of the session, like user name, email, chat id etc.
# 2. Connection string - a string that specifies the database connection. For SQLite, that string is `slqlite:///` followed by the name of the database file.  If that file doesn't exist, it will be created.

# In[1]:


from langchain_community.chat_message_histories import SQLChatMessageHistory

chat_message_history = SQLChatMessageHistory(
    session_id="test_session_id", connection_string="sqlite:///sqlite.db"
)

chat_message_history.add_user_message("Hello")
chat_message_history.add_ai_message("Hi")


# In[2]:


chat_message_history.messages


# ## Chaining
#
# We can easily combine this message history class with [LCEL Runnables](/docs/how_to/message_history)
#
# To do this we will want to use OpenAI, so we need to install that.  We will also need to set the OPENAI_API_KEY environment variable to your OpenAI key.
#
# ```bash
# pip install -U langchain-openai
#
# export OPENAI_API_KEY='sk-xxxxxxx'
# ```

# In[3]:


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


# In[4]:


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | ChatOpenAI()


# In[5]:


chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///sqlite.db"
    ),
    input_messages_key="question",
    history_messages_key="history",
)


# In[8]:


# This is where we configure the session id
config = {"configurable": {"session_id": "<SQL_SESSION_ID>"}}


# In[9]:


chain_with_history.invoke({"question": "Hi! I'm bob"}, config=config)


# In[10]:


chain_with_history.invoke({"question": "Whats my name"}, config=config)
