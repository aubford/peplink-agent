#!/usr/bin/env python
# coding: utf-8

# # Spark SQL Toolkit
#
# This notebook shows how to use agents to interact with `Spark SQL`. Similar to [SQL Database Agent](/docs/integrations/tools/sql_database), it is designed to address general inquiries about `Spark SQL` and facilitate error recovery.
#
# **NOTE: Note that, as this agent is in active development, all answers might not be correct. Additionally, it is not guaranteed that the agent won't perform DML statements on your Spark cluster given certain questions. Be careful running it on sensitive data!**

# ## Initialization

# In[1]:


from langchain_community.agent_toolkits import SparkSQLToolkit, create_spark_sql_agent
from langchain_community.utilities.spark_sql import SparkSQL
from langchain_openai import ChatOpenAI


# In[2]:


from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
schema = "langchain_example"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {schema}")
spark.sql(f"USE {schema}")
csv_file_path = "titanic.csv"
table = "titanic"
spark.read.csv(csv_file_path, header=True, inferSchema=True).write.saveAsTable(table)
spark.table(table).show()


# In[3]:


# Note, you can also connect to Spark via Spark connect. For example:
# db = SparkSQL.from_uri("sc://localhost:15002", schema=schema)
spark_sql = SparkSQL(schema=schema)
llm = ChatOpenAI(temperature=0)
toolkit = SparkSQLToolkit(db=spark_sql, llm=llm)
agent_executor = create_spark_sql_agent(llm=llm, toolkit=toolkit, verbose=True)


# ## Example: describing a table

# In[4]:


agent_executor.run("Describe the titanic table")


# ## Example: running queries

# In[5]:


agent_executor.run("whats the square root of the average age?")


# In[6]:


agent_executor.run("What's the name of the oldest survived passenger?")
