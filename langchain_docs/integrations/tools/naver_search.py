#!/usr/bin/env python
# coding: utf-8

# # Naver Search
# 
# ## Overview
# 
# ### Integration details
# 
# | Class | Package | Serializable | JS support |  Package latest |
# | :--- | :--- | :---: | :---: | :---: |
# | NaverSearchResults | [langchain-naver-community](https://pypi.org/project/langchain-naver-community/) | ❌ | ❌ |  ![PyPI - Version](https://img.shields.io/pypi/v/langchain-naver-community?style=flat-square&label=%20) |
# 
# ### Tool features
# 
# **Search** : The Naver Search Tool provides a simple interface to search Naver and get results.
# 
# ## Setup
# ### Setting Up API Credentials
# To use Naver Search, you need to obtain API credentials. Follow these steps:
# 
# Sign in to the [Naver Developers portal](https://developers.naver.com/main/).
# Create a new application and enable the Search API.
# Obtain your **NAVER_CLIENT_ID** and **NAVER_CLIENT_SECRET** from the "Application List" section.
# 
# ### Setting Up Environment Variables
# After obtaining the credentials, set them as environment variables in your script:

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  langchain-naver-community')


# In[2]:


import getpass
import os

if not os.environ.get("NAVER_CLIENT_ID"):
    os.environ["NAVER_CLIENT_ID"] = getpass.getpass("Enter your Naver Client ID:\n")

if not os.environ.get("NAVER_CLIENT_SECRET"):
    os.environ["NAVER_CLIENT_SECRET"] = getpass.getpass(
        "Enter your Naver Client Secret:\n"
    )


# ## Instantiation

# In[ ]:


from langchain_naver_community.utils import NaverSearchAPIWrapper

search = NaverSearchAPIWrapper()


# ## Invocation

# In[5]:


search.results("Seoul")[:3]


# ## Tool Usage

# In[19]:


from langchain_naver_community.tool import NaverSearchResults
from langchain_naver_community.utils import NaverSearchAPIWrapper

search = NaverSearchAPIWrapper()

tool = NaverSearchResults(api_wrapper=search)

tool.invoke("what is the weather in seoul?")[3:5]


# ## Use within an agent
# 
# The Naver Search tool can be integrated into LangChain agents for more complex tasks. Below we demonstrate how to set up an agent that can search Naver for current information.
# 

# In[ ]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

system_prompt = """
You are a helpful assistant that can search the web for information.
"""


# In[ ]:


from langchain_naver_community.tool import NaverNewsSearch
from langgraph.prebuilt import create_react_agent

tools = [NaverNewsSearch()]

agent_executor = create_react_agent(
    llm,
    tools,
    prompt=system_prompt,
)


# Now we can run the agent with a query.

# In[ ]:


query = "What is the weather in Seoul?"
result = agent_executor.invoke({"messages": [("human", query)]})
result["messages"][-1].content


# ## API reference
# 
# 
