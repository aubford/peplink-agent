#!/usr/bin/env python
# coding: utf-8

# # Exa Search

# Exa is a search engine fully designed for use by LLMs. Search for documents on the internet using **natural language queries**, then retrieve **cleaned HTML content** from desired documents.
# 
# Unlike keyword-based search (Google), Exa's neural search capabilities allow it to semantically understand queries and return relevant documents. For example, we could search `"fascinating article about cats"` and compare the search results from [Google](https://www.google.com/search?q=fascinating+article+about+cats) and [Exa](https://search.exa.ai/search?q=fascinating%20article%20about%20cats&autopromptString=Here%20is%20a%20fascinating%20article%20about%20cats%3A). Google gives us SEO-optimized listicles based on the keyword "fascinating". Exa just works.
# 
# This notebook goes over how to use Exa Search with LangChain.
# 
# First, get an Exa API key and add it as an environment variable. Get $10 free credit (plus more by completing certain actions like making your first search) by [signing up here](https://dashboard.exa.ai/).

# In[1]:


import os

api_key = os.getenv("EXA_API_KEY")  # Set your API key as an environment variable


# And install the integration package

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet langchain-exa')

# and some deps for this notebook
get_ipython().run_line_magic('pip', 'install --upgrade --quiet langchain langchain-openai langchain-community')


# ## Using ExaSearchRetriever
# 
# ExaSearchRetriever is a retriever that uses Exa Search to retrieve relevant documents.

# :::note
# 
# The `max_characters` parameter for **TextContentsOptions** used to be called `max_length` which is now deprecated. Make sure to use `max_characters` instead.
# 
# :::

# ## Using the Exa SDK as LangChain Agent Tools
# 
# The [Exa SDK](https://docs.exa.ai/) creates a client that can interact with three main Exa API endpoints:
# 
# - `search`: Given a natural language search query, retrieve a list of search results.
# - `find_similar`: Given a URL, retrieve a list of search results corresponding to webpages which are similar to the document at the provided URL.
# - `get_contents`: Given a list of document ids fetched from `search` or `find_similar`, get cleaned HTML content for each document.
# 
# The `exa_py` SDK combines these endpoints into two powerful calls. Using these provide the most flexible and efficient use cases of Exa search:
# 
# 1. `search_and_contents`: Combines the `search` and `get_contents` endpoints to retrieve search results along with their content in a single operation.
# 2. `find_similar_and_contents`: Combines the `find_similar` and `get_contents` endpoints to find similar pages and retrieve their content in one call.
# 
# We can use the `@tool` decorator and docstrings to create LangChain Tool wrappers that tell an LLM agent how to use these combined Exa functionalities effectively. This approach simplifies usage and reduces the number of API calls needed to get comprehensive results.
# 
# Before writing code, ensure you have `langchain-exa` installed

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  langchain-exa')


# In[5]:


import os

from exa_py import Exa
from langchain_core.tools import tool

exa = Exa(api_key=os.environ["EXA_API_KEY"])


@tool
def search_and_contents(query: str):
    """Search for webpages based on the query and retrieve their contents."""
    # This combines two API endpoints: search and contents retrieval
    return exa.search_and_contents(
        query, use_autoprompt=True, num_results=5, text=True, highlights=True
    )


@tool
def find_similar_and_contents(url: str):
    """Search for webpages similar to a given URL and retrieve their contents.
    The url passed in should be a URL returned from `search_and_contents`.
    """
    # This combines two API endpoints: find similar and contents retrieval
    return exa.find_similar_and_contents(url, num_results=5, text=True, highlights=True)


tools = [search_and_contents, find_similar_and_contents]


# ### Providing Exa Tools to an Agent
# 
# We can provide the Exa tools we just created to a LangChain `OpenAIFunctionsAgent`. When asked to `Summarize for me a fascinating article about cats`, the agent uses the `search` tool to perform a Exa search with an appropriate search query, uses the `get_contents` tool to perform Exa content retrieval, and then returns a summary of the retrieved content.

# In[ ]:


from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

system_message = SystemMessage(
    content="You are a web researcher who answers user questions by looking up information on the internet and retrieving contents of helpful documents. Cite your sources."
)

agent_prompt = OpenAIFunctionsAgent.create_prompt(system_message)
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# In[16]:


agent_executor.run("Summarize for me a fascinating article about cats.")


# ## Advanced Exa Features
# 
# Exa supports powerful filters by domain and date. We can provide a more powerful `search` tool to the agent that lets it decide to apply filters if they are useful for the objective. See all of Exa's search features [here](https://github.com/metaphorsystems/metaphor-python/).
# 
# [//]: # "TODO(erick): switch metaphor github link to exa github link when sdk published"

# In[11]:


import os

from exa_py import Exa
from langchain_core.tools import tool

exa = Exa(api_key=os.environ["EXA_API_KEY"])


@tool
def search_and_contents(
    query: str,
    include_domains: list[str] = None,
    exclude_domains: list[str] = None,
    start_published_date: str = None,
    end_published_date: str = None,
    include_text: list[str] = None,
    exclude_text: list[str] = None,
):
    """
    Search for webpages based on the query and retrieve their contents.

    Parameters:
    - query (str): The search query.
    - include_domains (list[str], optional): Restrict the search to these domains.
    - exclude_domains (list[str], optional): Exclude these domains from the search.
    - start_published_date (str, optional): Restrict to documents published after this date (YYYY-MM-DD).
    - end_published_date (str, optional): Restrict to documents published before this date (YYYY-MM-DD).
    - include_text (list[str], optional): Only include results containing these phrases.
    - exclude_text (list[str], optional): Exclude results containing these phrases.
    """
    return exa.search_and_contents(
        query,
        use_autoprompt=True,
        num_results=5,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        start_published_date=start_published_date,
        end_published_date=end_published_date,
        include_text=include_text,
        exclude_text=exclude_text,
        text=True,
        highlights=True,
    )


@tool
def find_similar_and_contents(
    url: str,
    exclude_source_domain: bool = False,
    start_published_date: str = None,
    end_published_date: str = None,
):
    """
    Search for webpages similar to a given URL and retrieve their contents.
    The url passed in should be a URL returned from `search_and_contents`.

    Parameters:
    - url (str): The URL to find similar pages for.
    - exclude_source_domain (bool, optional): If True, exclude pages from the same domain as the source URL.
    - start_published_date (str, optional): Restrict to documents published after this date (YYYY-MM-DD).
    - end_published_date (str, optional): Restrict to documents published before this date (YYYY-MM-DD).
    """
    return exa.find_similar_and_contents(
        url,
        num_results=5,
        exclude_source_domain=exclude_source_domain,
        start_published_date=start_published_date,
        end_published_date=end_published_date,
        text=True,
        highlights={"num_sentences": 1, "highlights_per_url": 1},
    )


tools = [search_and_contents, find_similar_and_contents]


# Now we ask the agent to summarize an article with constraints on domain and publish date. We will use a GPT-4 agent for extra powerful reasoning capability to support more complex tool usage.
# 
# The agent correctly uses the search filters to find an article with the desired constraints, and once again retrieves the content and returns a summary.

# In[14]:


from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="gpt-4o")

system_message = SystemMessage(
    content="You are a web researcher who answers user questions by looking up information on the internet and retrieving contents of helpful documents. Cite your sources."
)

agent_prompt = OpenAIFunctionsAgent.create_prompt(system_message)
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# In[15]:


agent_executor.run(
    "Summarize for me an interesting article about AI from lesswrong.com published after October 2023."
)

