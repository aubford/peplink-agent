#!/usr/bin/env python
# coding: utf-8

# # Databricks Unity Catalog (UC)
# 
# This notebook shows how to use UC functions as LangChain tools, with both LangChain and LangGraph agent APIs.
# 
# See Databricks documentation ([AWS](https://docs.databricks.com/en/sql/language-manual/sql-ref-syntax-ddl-create-sql-function.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/sql-ref-syntax-ddl-create-sql-function)|[GCP](https://docs.gcp.databricks.com/en/sql/language-manual/sql-ref-syntax-ddl-create-sql-function.html)) to learn how to create SQL or Python functions in UC. Do not skip function and parameter comments, which are critical for LLMs to call functions properly.
# 
# In this example notebook, we create a simple Python function that executes arbitrary code and use it as a LangChain tool:
# 
# ```sql
# CREATE FUNCTION main.tools.python_exec (
#   code STRING COMMENT 'Python code to execute. Remember to print the final result to stdout.'
# )
# RETURNS STRING
# LANGUAGE PYTHON
# COMMENT 'Executes Python code and returns its stdout.'
# AS $$
#   import sys
#   from io import StringIO
#   stdout = StringIO()
#   sys.stdout = stdout
#   exec(code)
#   return stdout.getvalue()
# $$
# ```
# 
# It runs in a secure and isolated environment within a Databricks SQL warehouse.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet databricks-sdk langchain-community databricks-langchain langgraph mlflow')


# In[ ]:


from databricks_langchain import ChatDatabricks

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct")


# In[3]:


from databricks_langchain.uc_ai import (
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)

client = DatabricksFunctionClient()
set_uc_function_client(client)

tools = UCFunctionToolkit(
    # Include functions as tools using their qualified names.
    # You can use "{catalog_name}.{schema_name}.*" to get all functions in a schema.
    function_names=["main.tools.python_exec"]
).tools


# (Optional) To increase the retry time for getting a function execution response, set environment variable UC_TOOL_CLIENT_EXECUTION_TIMEOUT. Default retry time value is 120s.
# ## LangGraph agent example

# In[ ]:


import os

os.environ["UC_TOOL_CLIENT_EXECUTION_TIMEOUT"] = "200"


# ## LangGraph agent example

# In[4]:


from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    llm,
    tools,
    prompt="You are a helpful assistant. Make sure to use tool for information.",
)
agent.invoke({"messages": [{"role": "user", "content": "36939 * 8922.4"}]})


# ## LangChain agent example

# In[5]:


from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Make sure to use tool for information.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)


# In[6]:


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "36939 * 8922.4"})

