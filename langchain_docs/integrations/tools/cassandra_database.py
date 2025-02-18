#!/usr/bin/env python
# coding: utf-8

# # Cassandra Database Toolkit
#
# >`Apache Cassandra®` is a widely used database for storing transactional application data. The introduction of functions and >tooling in Large Language Models has opened up some exciting use cases for existing data in Generative AI applications.
#
# >The `Cassandra Database` toolkit enables AI engineers to integrate agents with Cassandra data efficiently, offering
# >the following features:
# > - Fast data access through optimized queries. Most queries should run in single-digit ms or less.
# > - Schema introspection to enhance LLM reasoning capabilities
# > - Compatibility with various Cassandra deployments, including Apache Cassandra®, DataStax Enterprise™, and DataStax Astra™
# > - Currently, the toolkit is limited to SELECT queries and schema introspection operations. (Safety first)
#
# For more information on creating a Cassandra DB agent see the [CQL agent cookbook](https://github.com/langchain-ai/langchain/blob/master/cookbook/cql_agent.ipynb)
#
# ## Quick Start
#  - Install the `cassio` library
#  - Set environment variables for the Cassandra database you are connecting to
#  - Initialize `CassandraDatabase`
#  - Pass the tools to your agent with `toolkit.get_tools()`
#  - Sit back and watch it do all your work for you
#
# ## Theory of Operation
#
# `Cassandra Query Language (CQL)` is the primary *human-centric* way of interacting with a Cassandra database. While offering some flexibility when generating queries, it requires knowledge of Cassandra data modeling best practices. LLM function calling gives an agent the ability to reason and then choose a tool to satisfy the request. Agents using LLMs should reason using Cassandra-specific logic when choosing the appropriate toolkit or chain of toolkits. This reduces the randomness introduced when LLMs are forced to provide a top-down solution. Do you want an LLM to have complete unfettered access to your database? Yeah. Probably not. To accomplish this, we provide a prompt for use when constructing questions for the agent:
#
# You are an Apache Cassandra expert query analysis bot with the following features
# and rules:
#  - You will take a question from the end user about finding specific
#    data in the database.
#  - You will examine the schema of the database and create a query path.
#  - You will provide the user with the correct query to find the data they are looking
#    for, showing the steps provided by the query path.
#  - You will use best practices for querying Apache Cassandra using partition keys
#    and clustering columns.
#  - Avoid using ALLOW FILTERING in the query.
#  - The goal is to find a query path, so it may take querying other tables to get
#    to the final answer.
#
# The following is an example of a query path in JSON format:
#
# ```json
#  {
#   "query_paths": [
#     {
#       "description": "Direct query to users table using email",
#       "steps": [
#         {
#           "table": "user_credentials",
#           "query":
#              "SELECT userid FROM user_credentials WHERE email = 'example@example.com';"
#         },
#         {
#           "table": "users",
#           "query": "SELECT * FROM users WHERE userid = ?;"
#         }
#       ]
#     }
#   ]
# }
# ```
#
# ## Tools Provided
#
# ### `cassandra_db_schema`
# Gathers all schema information for the connected database or a specific schema. Critical for the agent when determining actions.
#
# ### `cassandra_db_select_table_data`
# Selects data from a specific keyspace and table. The agent can pass paramaters for a predicate and limits on the number of returned records.
#
# ### `cassandra_db_query`
# Expiriemental alternative to `cassandra_db_select_table_data` which takes a query string completely formed by the agent instead of parameters. *Warning*: This can lead to unusual queries that may not be as performant(or even work). This may be removed in future releases. If it does something cool, we want to know about that too. You never know!

# ## Environment Setup
#
# Install the following Python modules:
#
# ```bash
# pip install ipykernel python-dotenv cassio langchain_openai langchain langchain-community langchainhub
# ```
#
# ### .env file
# Connection is via `cassio` using `auto=True` parameter, and the notebook uses OpenAI. You should create a `.env` file accordingly.
#
# For Casssandra, set:
# ```bash
# CASSANDRA_CONTACT_POINTS
# CASSANDRA_USERNAME
# CASSANDRA_PASSWORD
# CASSANDRA_KEYSPACE
# ```
#
# For Astra, set:
# ```bash
# ASTRA_DB_APPLICATION_TOKEN
# ASTRA_DB_DATABASE_ID
# ASTRA_DB_KEYSPACE
# ```
#
# For example:
#
# ```bash
# # Connection to Astra:
# ASTRA_DB_DATABASE_ID=a1b2c3d4-...
# ASTRA_DB_APPLICATION_TOKEN=AstraCS:...
# ASTRA_DB_KEYSPACE=notebooks
#
# # Also set
# OPENAI_API_KEY=sk-....
# ```
#
# (You may also modify the below code to directly connect with `cassio`.)

# In[ ]:


from dotenv import load_dotenv

load_dotenv(override=True)


# In[2]:


# Import necessary libraries
import os

import cassio
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.agent_toolkits.cassandra_database.toolkit import (
    CassandraDatabaseToolkit,
)
from langchain_community.tools.cassandra_database.prompt import QUERY_PATH_PROMPT
from langchain_community.utilities.cassandra_database import CassandraDatabase
from langchain_openai import ChatOpenAI


# ## Connect to a Cassandra Database

# In[3]:


cassio.init(auto=True)
session = cassio.config.resolve_session()
if not session:
    raise Exception(
        "Check environment configuration or manually configure cassio connection parameters"
    )


# In[4]:


# Test data pep

session = cassio.config.resolve_session()

session.execute("""DROP KEYSPACE IF EXISTS langchain_agent_test; """)

session.execute(
    """
CREATE KEYSPACE if not exists langchain_agent_test 
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};
"""
)

session.execute(
    """
    CREATE TABLE IF NOT EXISTS langchain_agent_test.user_credentials (
    user_email text PRIMARY KEY,
    user_id UUID,
    password TEXT
);
"""
)

session.execute(
    """
    CREATE TABLE IF NOT EXISTS langchain_agent_test.users (
    id UUID PRIMARY KEY,
    name TEXT,
    email TEXT
);"""
)

session.execute(
    """
    CREATE TABLE IF NOT EXISTS langchain_agent_test.user_videos ( 
    user_id UUID,
    video_id UUID,
    title TEXT,
    description TEXT,
    PRIMARY KEY (user_id, video_id)
);
"""
)

user_id = "522b1fe2-2e36-4cef-a667-cd4237d08b89"
video_id = "27066014-bad7-9f58-5a30-f63fe03718f6"

session.execute(
    f"""
    INSERT INTO langchain_agent_test.user_credentials (user_id, user_email) 
    VALUES ({user_id}, 'patrick@datastax.com');
"""
)

session.execute(
    f"""
    INSERT INTO langchain_agent_test.users (id, name, email) 
    VALUES ({user_id}, 'Patrick McFadin', 'patrick@datastax.com');
"""
)

session.execute(
    f"""
    INSERT INTO langchain_agent_test.user_videos (user_id, video_id, title)
    VALUES ({user_id}, {video_id}, 'Use Langflow to Build a LangChain LLM Application in 5 Minutes');
"""
)

session.set_keyspace("langchain_agent_test")


# In[5]:


# Create a CassandraDatabase instance
# Uses the cassio session to connect to the database
db = CassandraDatabase()


# In[6]:


# Choose the LLM that will drive the agent
# Only certain models support this
llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
toolkit = CassandraDatabaseToolkit(db=db)

tools = toolkit.get_tools()

print("Available tools:")
for tool in tools:
    print(tool.name + "\t- " + tool.description)


# In[7]:


prompt = hub.pull("hwchase17/openai-tools-agent")

# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt)


# In[9]:


input = (
    QUERY_PATH_PROMPT
    + "\n\nHere is your task: Find all the videos that the user with the email address 'patrick@datastax.com' has uploaded to the langchain_agent_test keyspace."
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({"input": input})

print(response["output"])
