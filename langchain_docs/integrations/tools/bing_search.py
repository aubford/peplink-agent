#!/usr/bin/env python
# coding: utf-8

# # Bing Search
#
# > [Bing Search](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/) is an Azure service and enables safe, ad-free, location-aware search results, surfacing relevant information from billions of web documents. Help your users find what they're looking for from the world-wide-web by harnessing Bing's ability to comb billions of webpages, images, videos, and news with a single API call.

# ## Setup
# Following the [instruction](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/create-bing-search-service-resource) to create Azure Bing Search v7 service, and get the subscription key
#
# The integration lives in the `langchain-community` package.

# In[ ]:


get_ipython().run_line_magic("pip", "install -U langchain-community")


# In[12]:


import getpass
import os

os.environ["BING_SUBSCRIPTION_KEY"] = getpass.getpass()
os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"


# In[3]:


from langchain_community.utilities import BingSearchAPIWrapper


# In[4]:


search = BingSearchAPIWrapper(k=4)


# In[18]:


search.run("python")


# ## Number of results
# You can use the `k` parameter to set the number of results

# In[6]:


search = BingSearchAPIWrapper(k=1)


# In[7]:


search.run("python")


# ## Metadata Results

# Run query through BingSearch and return snippet, title, and link metadata.
#
# - Snippet: The description of the result.
# - Title: The title of the result.
# - Link: The link to the result.

# In[8]:


search = BingSearchAPIWrapper()


# In[9]:


search.results("apples", 5)


# ## Tool Usage

# In[13]:


import os

from langchain_community.tools.bing_search import BingSearchResults
from langchain_community.utilities import BingSearchAPIWrapper

api_wrapper = BingSearchAPIWrapper()
tool = BingSearchResults(api_wrapper=api_wrapper)
tool


# In[14]:


import json

# .invoke wraps utility.results
response = tool.invoke("What is the weather in Shanghai?")
response = json.loads(response.replace("'", '"'))
for item in response:
    print(item)


# ## Chaining
#
# We show here how to use it as part of an [agent](/docs/tutorials/agents). We use the OpenAI Functions Agent, so we will need to setup and install the required dependencies for that. We will also use [LangSmith Hub](https://smith.langchain.com/hub) to pull the prompt from, so we will need to install that.

# In[ ]:


# you need a model to use in the chain
get_ipython().run_line_magic(
    "pip",
    "install --upgrade --quiet langchain langchain-openai langchainhub langchain-community",
)


# In[11]:


import getpass
import os

from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import AzureChatOpenAI

os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass()
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://<your-endpoint>.openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-06-01-preview"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "<your-deployment-name>"

instructions = """You are an assistant."""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)
llm = AzureChatOpenAI(
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
tool = BingSearchResults(api_wrapper=api_wrapper)
tools = [tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)
agent_executor.invoke({"input": "What happened in the latest burning man floods?"})
