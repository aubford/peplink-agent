#!/usr/bin/env python
# coding: utf-8

# # Databricks
# 
# This notebook covers how to connect to the [Databricks runtimes](https://docs.databricks.com/runtime/index.html) and [Databricks SQL](https://www.databricks.com/product/databricks-sql) using the SQLDatabase wrapper of LangChain.
# It is broken into 3 parts: installation and setup, connecting to Databricks, and examples.

# ## Installation and Setup

# In[1]:


get_ipython().system('pip install databricks-sql-connector')


# ## Connecting to Databricks
# 
# You can connect to [Databricks runtimes](https://docs.databricks.com/runtime/index.html) and [Databricks SQL](https://www.databricks.com/product/databricks-sql) using the `SQLDatabase.from_databricks()` method.
# 
# ### Syntax
# ```python
# SQLDatabase.from_databricks(
#     catalog: str,
#     schema: str,
#     host: Optional[str] = None,
#     api_token: Optional[str] = None,
#     warehouse_id: Optional[str] = None,
#     cluster_id: Optional[str] = None,
#     engine_args: Optional[dict] = None,
#     **kwargs: Any)
# ```
# ### Required Parameters
# * `catalog`: The catalog name in the Databricks database.
# * `schema`: The schema name in the catalog.
# 
# ### Optional Parameters
# There following parameters are optional. When executing the method in a Databricks notebook, you don't need to provide them in most of the cases.
# * `host`: The Databricks workspace hostname, excluding 'https://' part. Defaults to 'DATABRICKS_HOST' environment variable or current workspace if in a Databricks notebook.
# * `api_token`: The Databricks personal access token for accessing the Databricks SQL warehouse or the cluster. Defaults to 'DATABRICKS_TOKEN' environment variable or a temporary one is generated if in a Databricks notebook.
# * `warehouse_id`: The warehouse ID in the Databricks SQL.
# * `cluster_id`: The cluster ID in the Databricks Runtime. If running in a Databricks notebook and both 'warehouse_id' and 'cluster_id' are None, it uses the ID of the cluster the notebook is attached to.
# * `engine_args`: The arguments to be used when connecting Databricks.
# * `**kwargs`: Additional keyword arguments for the `SQLDatabase.from_uri` method.

# ## Examples

# In[2]:


# Connecting to Databricks with SQLDatabase wrapper
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_databricks(catalog="samples", schema="nyctaxi")


# In[3]:


# Creating a OpenAI Chat LLM wrapper
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0, model_name="gpt-4")


# ### SQL Chain example
# 
# This example demonstrates the use of the [SQL Chain](https://python.langchain.com/en/latest/modules/chains/examples/sqlite.html) for answering a question over a Databricks database.

# In[4]:


from langchain_community.utilities import SQLDatabaseChain

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)


# In[5]:


db_chain.run(
    "What is the average duration of taxi rides that start between midnight and 6am?"
)


# ### SQL Database Agent example
# 
# This example demonstrates the use of the [SQL Database Agent](/docs/integrations/tools/sql_database) for answering questions over a Databricks database.

# In[7]:


from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)


# In[8]:


agent.run("What is the longest trip distance and how long did it take?")

