#!/usr/bin/env python
# coding: utf-8

# # Polygon IO Toolkit and Tools
# 
# This notebook shows how to use agents to interact with the [Polygon IO](https://polygon.io/) toolkit. The toolkit provides access to Polygon's Stock Market Data API.

# ## Setup
# 
# ### Installation
# 
# To use Polygon IO tools, you need to install the `langchain-community` package.

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-community > /dev/null')


# ### Credentials
# 
# Get your Polygon IO API key [here](https://polygon.io/), and then set it below. 

# In[2]:


import getpass
import os

if "POLYGON_API_KEY" not in os.environ:
    os.environ["POLYGON_API_KEY"] = getpass.getpass()


# It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability

# In[3]:


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()


# ## Toolkit
# 
# Polygon IO provides both a toolkit and individual tools for each of the tools included in the toolkit. Let's first explore using the toolkit and then we will walk through using the individual tools.
# 
# ### Initialization
# 
# We can initialize the toolkit by importing it alongside the API wrapper needed to use the tools.

# In[4]:


from langchain_community.agent_toolkits.polygon.toolkit import PolygonToolkit
from langchain_community.utilities.polygon import PolygonAPIWrapper

polygon = PolygonAPIWrapper()
toolkit = PolygonToolkit.from_polygon_api_wrapper(polygon)


# ### Tools
# 
# We can examine the tools included in this toolkit:

# In[5]:


toolkit.get_tools()


# ### Use within an agent
# 
# Next we can add our toolkit to an agent and use it!

# In[11]:


from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="gpt-4o")

instructions = """You are an assistant."""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)

agent = create_openai_functions_agent(llm, toolkit.get_tools(), prompt)


# In[12]:


agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
)


# We can examine yesterdays information for a certain ticker:

# In[13]:


agent_executor.invoke({"input": "What was yesterdays financial info for AAPL?"})


# We can also ask for recent news regarding a stock:

# In[14]:


agent_executor.invoke({"input": "What is the recent new regarding MSFT?"})


# You can also ask about financial information for a company:

# In[15]:


agent_executor.invoke(
    {"input": "What were last quarters financial numbers for Nvidia?"}
)


# Lastly, you can get live data, although this requires a "Stocks Advanced" subscription

# In[ ]:


agent_executor.invoke({"input": "What is Doordash stock price right now?"})


# ### API reference
# 
# For detailed documentation of all the Polygon IO toolkit features and configurations head to the API reference: https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.polygon.toolkit.PolygonToolkit.html

# ## Tools
# 
# First, let's set up the API wrapper that we will use for all the tools and then we will walk through each one of them.

# In[16]:


from langchain_community.utilities.polygon import PolygonAPIWrapper

api_wrapper = PolygonAPIWrapper()


# ### Aggregate
# 
# This tool shows aggregate information for a stock.

# In[25]:


from langchain_community.tools.polygon.aggregates import PolygonAggregates

aggregate_tool = PolygonAggregates(api_wrapper=api_wrapper)

# We can invoke directly with input
res = aggregate_tool.invoke(
    {
        "ticker": "AAPL",
        "timespan": "day",
        "timespan_multiplier": 1,
        "from_date": "2024-08-01",
        "to_date": "2024-08-07",
    }
)

model_generated_tool_call = {
    "args": {
        "ticker": "AAPL",
        "timespan": "day",
        "timespan_multiplier": 1,
        "from_date": "2024-08-01",
        "to_date": "2024-08-07",
    },
    "id": "1",
    "name": aggregate_tool.name,
    "type": "tool_call",
}

# Or we can invoke with a tool call
res = aggregate_tool.invoke(model_generated_tool_call)

print(res)


# ### Financials
# 
# This tool provides general financial information about a stock

# In[35]:


from langchain_community.tools.polygon.financials import PolygonFinancials

financials_tool = PolygonFinancials(api_wrapper=api_wrapper)

# We can invoke directly with input
res = financials_tool.invoke({"query": "AAPL"})

model_generated_tool_call = {
    "args": {"query": "AAPL"},
    "id": "1",
    "name": financials_tool.name,
    "type": "tool_call",
}

# Or we can invoke with a tool call
res = financials_tool.invoke(model_generated_tool_call)

print(res)


# ### Last Quote
# 
# This tool provides information about the live data of a stock, although it requires a "Stocks Advanced" subscription to use.

# In[ ]:


from langchain_community.tools.polygon.last_quote import PolygonLastQuote

last_quote_tool = PolygonLastQuote(api_wrapper=api_wrapper)

# We can invoke directly with input
res = last_quote_tool.invoke({"query": "AAPL"})

model_generated_tool_call = {
    "args": {"query": "AAPL"},
    "id": "1",
    "name": last_quote_tool.name,
    "type": "tool_call",
}

# Or we can invoke with a tool call
res = last_quote_tool.invoke(model_generated_tool_call)


# ### Ticker News
# 
# This tool provides recent news about a certain ticker.

# In[33]:


from langchain_community.tools.polygon.ticker_news import PolygonTickerNews

news_tool = PolygonTickerNews(api_wrapper=api_wrapper)

# We can invoke directly with input
res = news_tool.invoke({"query": "AAPL"})

model_generated_tool_call = {
    "args": {"query": "AAPL"},
    "id": "1",
    "name": news_tool.name,
    "type": "tool_call",
}

# Or we can invoke with a tool call
res = news_tool.invoke(model_generated_tool_call)

print(res)


# ### API reference
# 
# For detailed documentation of all Polygon IO tools head to the API reference for each:
# 
# - Aggregate: https://python.langchain.com/api_reference/community/tools/langchain_community.tools.polygon.aggregates.PolygonAggregates.html
# - Financials: https://python.langchain.com/api_reference/community/tools/langchain_community.tools.polygon.financials.PolygonFinancials.html
# - Last Quote: https://python.langchain.com/api_reference/community/tools/langchain_community.tools.polygon.last_quote.PolygonLastQuote.html
# - Ticker News: https://python.langchain.com/api_reference/community/tools/langchain_community.tools.polygon.ticker_news.PolygonTickerNews.html
