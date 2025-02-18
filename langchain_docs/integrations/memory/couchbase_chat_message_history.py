#!/usr/bin/env python
# coding: utf-8

# # Couchbase
# > Couchbase is an award-winning distributed NoSQL cloud database that delivers unmatched versatility, performance, scalability, and financial value for all of your cloud, mobile, AI, and edge computing applications. Couchbase embraces AI with coding assistance for developers and vector search for their applications.
#
# This notebook goes over how to use the `CouchbaseChatMessageHistory` class to store the chat message history in a Couchbase cluster
#

# ## Set Up Couchbase Cluster
# To run this demo, you need a Couchbase Cluster.
#
# You can work with both [Couchbase Capella](https://www.couchbase.com/products/capella/) and your self-managed Couchbase Server.

# ## Install Dependencies
# `CouchbaseChatMessageHistory` lives inside the `langchain-couchbase` package.

# In[1]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet langchain-couchbase")


# ## Create Couchbase Connection Object
# We create a connection to the Couchbase cluster initially and then pass the cluster object to the Vector Store.
#
# Here, we are connecting using the username and password. You can also connect using any other supported way to your cluster.
#
# For more information on connecting to the Couchbase cluster, please check the [Python SDK documentation](https://docs.couchbase.com/python-sdk/current/hello-world/start-using-sdk.html#connect).

# In[2]:


COUCHBASE_CONNECTION_STRING = (
    "couchbase://localhost"  # or "couchbases://localhost" if using TLS
)
DB_USERNAME = "Administrator"
DB_PASSWORD = "Password"


# In[3]:


from datetime import timedelta

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions

auth = PasswordAuthenticator(DB_USERNAME, DB_PASSWORD)
options = ClusterOptions(auth)
cluster = Cluster(COUCHBASE_CONNECTION_STRING, options)

# Wait until the cluster is ready for use.
cluster.wait_until_ready(timedelta(seconds=5))


# We will now set the bucket, scope, and collection names in the Couchbase cluster that we want to use for storing the message history.
#
# Note that the bucket, scope, and collection need to exist before using them to store the message history.

# In[4]:


BUCKET_NAME = "langchain-testing"
SCOPE_NAME = "_default"
COLLECTION_NAME = "conversational_cache"


# ## Usage
# In order to store the messages, you need the following:
# - Couchbase Cluster object: Valid connection to the Couchbase cluster
# - bucket_name: Bucket in cluster to store the chat message history
# - scope_name: Scope in bucket to store the message history
# - collection_name: Collection in scope to store the message history
# - session_id: Unique identifier for the session
#
# Optionally you can configure the following:
# - session_id_key: Field in the chat message documents to store the `session_id`
# - message_key: Field in the chat message documents to store the message content
# - create_index: Used to specify if the index needs to be created on the collection. By default, an index is created on the `message_key` and the `session_id_key` of the documents
# - ttl: Used to specify a time in `timedelta` to live for the documents after which they will get deleted automatically from the storage.

# In[5]:


from langchain_couchbase.chat_message_histories import CouchbaseChatMessageHistory

message_history = CouchbaseChatMessageHistory(
    cluster=cluster,
    bucket_name=BUCKET_NAME,
    scope_name=SCOPE_NAME,
    collection_name=COLLECTION_NAME,
    session_id="test-session",
)

message_history.add_user_message("hi!")

message_history.add_ai_message("how are you doing?")


# In[6]:


message_history.messages


# ## Specifying a Time to Live (TTL) for the Chat Messages
# The stored messages can be deleted after a specified time automatically by specifying a `ttl` parameter along with the initialization of the chat message history store.

# In[7]:


from langchain_couchbase.chat_message_histories import CouchbaseChatMessageHistory

message_history = CouchbaseChatMessageHistory(
    cluster=cluster,
    bucket_name=BUCKET_NAME,
    scope_name=SCOPE_NAME,
    collection_name=COLLECTION_NAME,
    session_id="test-session",
    ttl=timedelta(hours=24),
)


# ## Chaining
# The chat message history class can be used with [LCEL Runnables](https://python.langchain.com/docs/how_to/message_history/)

# In[8]:


import getpass
import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = getpass.getpass()


# In[9]:


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Create the LCEL runnable
chain = prompt | ChatOpenAI()


# In[10]:


chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: CouchbaseChatMessageHistory(
        cluster=cluster,
        bucket_name=BUCKET_NAME,
        scope_name=SCOPE_NAME,
        collection_name=COLLECTION_NAME,
        session_id=session_id,
    ),
    input_messages_key="question",
    history_messages_key="history",
)


# In[11]:


# This is where we configure the session id
config = {"configurable": {"session_id": "testing"}}


# In[12]:


chain_with_history.invoke({"question": "Hi! I'm bob"}, config=config)


# In[13]:


chain_with_history.invoke({"question": "Whats my name"}, config=config)
