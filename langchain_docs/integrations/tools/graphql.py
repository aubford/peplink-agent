#!/usr/bin/env python
# coding: utf-8

# # GraphQL
# 
# >[GraphQL](https://graphql.org/) is a query language for APIs and a runtime for executing those queries against your data. `GraphQL` provides a complete and understandable description of the data in your API, gives clients the power to ask for exactly what they need and nothing more, makes it easier to evolve APIs over time, and enables powerful developer tools.
# 
# By including a `BaseGraphQLTool` in the list of tools provided to an Agent, you can grant your Agent the ability to query data from GraphQL APIs for any purposes you need.
# 
# This Jupyter Notebook demonstrates how to use the `GraphQLAPIWrapper` component with an Agent.
# 
# In this example, we'll be using the public `Star Wars GraphQL API` available at the following endpoint: https://swapi-graphql.netlify.app/.netlify/functions/index.
# 
# First, you need to install `httpx` and `gql` Python packages.

# In[ ]:


pip install httpx gql > /dev/null


# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  langchain-community')


# Now, let's create a BaseGraphQLTool instance with the specified Star Wars API endpoint and initialize an Agent with the tool.

# In[3]:


from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)

tools = load_tools(
    ["graphql"],
    graphql_endpoint="https://swapi-graphql.netlify.app/.netlify/functions/index",
)

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


# Now, we can use the Agent to run queries against the Star Wars GraphQL API. Let's ask the Agent to list all the Star Wars films and their release dates.

# In[4]:


graphql_fields = """allFilms {
    films {
      title
      director
      releaseDate
      speciesConnection {
        species {
          name
          classification
          homeworld {
            name
          }
        }
      }
    }
  }

"""

suffix = "Search for the titles of all the stawars films stored in the graphql database that has this schema "


agent.run(suffix + graphql_fields)

