#!/usr/bin/env python
# coding: utf-8

# # AskNews

# > [AskNews](https://asknews.app) infuses any LLM with the latest global news (or historical news), using a single natural language query. Specifically, AskNews is enriching over 300k articles per day by translating, summarizing, extracting entities, and indexing them into hot and cold vector databases. AskNews puts these vector databases on a low-latency endpoint for you. When you query AskNews, you get back a prompt-optimized string that contains all the most pertinent enrichments (e.g. entities, classifications, translation, summarization). This means that you do not need to manage your own news RAG, and you do not need to worry about how to properly convey news information in a condensed way to your LLM.
# > AskNews is also committed to transparency, which is why our coverage is monitored and diversified across hundreds of countries, 13 languages, and 50 thousand sources. If you'd like to track our source coverage, you can visit our [transparency dashboard](https://asknews.app/en/transparency).
#
# ## Setup
#
# The integration lives in the `langchain-community` package. We also need to install the `asknews` package itself.
#
# ```bash
# pip install -U langchain-community asknews
# ```
#
# We also need to set our AskNews API credentials, which can be obtained at the [AskNews console](https://my.asknews.app).

# In[1]:


import getpass
import os

os.environ["ASKNEWS_CLIENT_ID"] = getpass.getpass()
os.environ["ASKNEWS_CLIENT_SECRET"] = getpass.getpass()


# ## Usage
#
# Here we show how to use the tool individually.

# In[11]:


from langchain_community.tools.asknews import AskNewsSearch

tool = AskNewsSearch(max_results=2)
tool.invoke({"query": "Effect of fed policy on tech sector"})


# ## Chaining
# We show here how to use it as part of an agent. We use the OpenAI Functions Agent, so we will need to setup and install the required dependencies for that. We will also use [LangSmith Hub](https://smith.langchain.com/hub) to pull the prompt from, so we will need to install that.
#
# ```bash
# pip install -U langchain-openai langchainhub
# ```

# In[4]:


import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()


# In[10]:


from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools.asknews import AskNewsSearch
from langchain_openai import ChatOpenAI

prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(temperature=0)
asknews_tool = AskNewsSearch()
tools = [asknews_tool]
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
agent_executor.invoke({"input": "How is the tech sector being affected by fed policy?"})


# In[ ]:
