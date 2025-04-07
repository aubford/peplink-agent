#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Memgraph
---
# # MemgraphToolkit
# 
# ## Overview
# 
# This will help you getting started with the Memgraph [toolkit](/docs/concepts/tools/#toolkits). 
# 
# Tools within `MemgraphToolkit` are designed for the interaction with the `Memgraph` database.
# 
# ## Setup
# 
# To be able tot follow the steps below, make sure you have a running Memgraph instance on your local host. For more details on how to run Memgraph, take a look at [Memgraph docs](https://memgraph.com/docs/getting-started)
#   

# If you want to get automated tracing from runs of individual tools, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[ ]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# This toolkit lives in the `langchain-memgraph` package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-memgraph')


# ## Instantiation
# 
# Now we can instantiate our toolkit:

# In[ ]:


from langchain.chat_models import init_chat_model
from langchain_memgraph import MemgraphToolkit
from langchain_memgraph.graphs.memgraph import Memgraph

db = Memgraph(url=url, username=username, password=password)

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

toolkit = MemgraphToolkit(
    db=db,  # Memgraph instance
    llm=llm,  # LLM chat model for LLM operations
)


# ## Tools
# 
# View available tools:

# In[ ]:


toolkit.get_tools()


# ## Invocation
# 
# Tools can be individually called by passing an arguments, for QueryMemgraphTool it would be: 
# 

# In[ ]:


from langchain_memgraph.tools import QueryMemgraphTool

# Rest of the code omitted for brevity

tool.invoke({QueryMemgraphTool({"query": "MATCH (n) RETURN n LIMIT 5"})})


# ## Use within an agent

# In[ ]:


from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(llm, tools)


# In[ ]:


example_query = "MATCH (n) RETURN n LIMIT 1"

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()


# ## API reference
# 
# For more details on API visit [Memgraph integration docs](https://memgraph.com/docs/ai-ecosystem/integrations#langchain)
