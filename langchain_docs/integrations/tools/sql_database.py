#!/usr/bin/env python
# coding: utf-8

# # SQLDatabase Toolkit
# 
# This will help you getting started with the SQL Database [toolkit](/docs/concepts/tools/#toolkits). For detailed documentation of all `SQLDatabaseToolkit` features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.sql.toolkit.SQLDatabaseToolkit.html).
# 
# Tools within the `SQLDatabaseToolkit` are designed to interact with a `SQL` database. 
# 
# A common application is to enable agents to answer questions using data in a relational database, potentially in an iterative fashion (e.g., recovering from errors).
# 
# **⚠️ Security note ⚠️**
# 
# Building Q&A systems of SQL databases requires executing model-generated SQL queries. There are inherent risks in doing this. Make sure that your database connection permissions are always scoped as narrowly as possible for your chain/agent's needs. This will mitigate though not eliminate the risks of building a model-driven system. For more on general security best practices, [see here](/docs/security).
# 
# ## Setup
# 
# To enable automated tracing of individual tools, set your [LangSmith](https://docs.smith.langchain.com/) API key:

# In[ ]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# This toolkit lives in the `langchain-community` package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  langchain-community')


# For demonstration purposes, we will access a prompt in the LangChain [Hub](https://smith.langchain.com/hub). We will also require `langgraph` to demonstrate the use of the toolkit with an agent. This is not required to use the toolkit.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet langchainhub langgraph')


# ## Instantiation
# 
# The `SQLDatabaseToolkit` toolkit requires:
# 
# - a [SQLDatabase](https://python.langchain.com/api_reference/community/utilities/langchain_community.utilities.sql_database.SQLDatabase.html) object;
# - a LLM or chat model (for instantiating the [QuerySQLCheckerTool](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.sql_database.tool.QuerySQLCheckerTool.html) tool).
# 
# Below, we instantiate the toolkit with these objects. Let's first create a database object.
# 
# This guide uses the example `Chinook` database based on [these instructions](https://database.guide/2-sample-databases-sqlite/).
# 
# Below we will use the `requests` library to pull the `.sql` file and create an in-memory SQLite database. Note that this approach is lightweight, but ephemeral and not thread-safe. If you'd prefer, you can follow the instructions to save the file locally as `Chinook.db` and instantiate the database via `db = SQLDatabase.from_uri("sqlite:///Chinook.db")`.

# In[1]:


import sqlite3

import requests
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


def get_engine_for_chinook_db():
    """Pull sql file, populate in-memory database, and create engine."""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


engine = get_engine_for_chinook_db()

db = SQLDatabase(engine)


# We will also need a LLM or chat model:
# 
# import ChatModelTabs from "@theme/ChatModelTabs";
# 
# <ChatModelTabs customVarName="llm" />
# 

# In[2]:


# | output: false
# | echo: false

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)


# We can now instantiate the toolkit:

# In[3]:


from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)


# ## Tools
# 
# View available tools:

# In[4]:


toolkit.get_tools()


# You can use the individual tools directly:

# In[ ]:


from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDatabaseTool,
)


# ## Use within an agent
# 
# Following the [SQL Q&A Tutorial](/docs/tutorials/sql_qa/#agents), below we equip a simple question-answering agent with the tools in our toolkit. First we pull a relevant prompt and populate it with its required parameters:

# In[6]:


from langchain import hub

prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

assert len(prompt_template.messages) == 1
print(prompt_template.input_variables)


# In[7]:


system_message = prompt_template.format(dialect="SQLite", top_k=5)


# We then instantiate the agent:

# In[8]:


from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(llm, toolkit.get_tools(), prompt=system_message)


# And issue it a query:

# In[9]:


example_query = "Which country's customers spent the most?"

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()


# We can also observe the agent recover from an error:

# In[10]:


example_query = "Who are the top 3 best selling artists?"

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()


# ## Specific functionality
# 
# `SQLDatabaseToolkit` implements a [.get_context](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.sql.toolkit.SQLDatabaseToolkit.html#langchain_community.agent_toolkits.sql.toolkit.SQLDatabaseToolkit.get_context) method as a convenience for use in prompts or other contexts.
# 
# **⚠️ Disclaimer ⚠️** : The agent may generate insert/update/delete queries. When this is not expected, use a custom prompt or create a SQL users without write permissions.
# 
# The final user might overload your SQL database by asking a simple question such as "run the biggest query possible". The generated query might look like:
# 
# ```sql
# SELECT * FROM "public"."users"
#     JOIN "public"."user_permissions" ON "public"."users".id = "public"."user_permissions".user_id
#     JOIN "public"."projects" ON "public"."users".id = "public"."projects".user_id
#     JOIN "public"."events" ON "public"."projects".id = "public"."events".project_id;
# ```
# 
# For a transactional SQL database, if one of the table above contains millions of rows, the query might cause trouble to other applications using the same database.
# 
# Most datawarehouse oriented databases support user-level quota, for limiting resource usage.

# ## API reference
# 
# For detailed documentation of all SQLDatabaseToolkit features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.sql.toolkit.SQLDatabaseToolkit.html).
