#!/usr/bin/env python
# coding: utf-8

# # ArXiv
#
# This notebook goes over how to use the `arxiv` tool with an agent.
#
# First, you need to install the `arxiv` python package.

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet  langchain-community arxiv"
)


# In[2]:


from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0.0)
tools = load_tools(
    ["arxiv"],
)
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# In[3]:


agent_executor.invoke(
    {
        "input": "What's the paper 1605.08386 about?",
    }
)


# ## The ArXiv API Wrapper
#
# The tool uses the `API Wrapper`. Below, we explore some of the features it provides.

# In[4]:


from langchain_community.utilities import ArxivAPIWrapper


# You can use the ArxivAPIWrapper to get information about a scientific article or articles. The query text is limited to 300 characters.
#
# The ArxivAPIWrapper returns these article fields:
# - Publishing date
# - Title
# - Authors
# - Summary
#
# The following query returns information about one article with the arxiv ID "1605.08386".

# In[5]:


arxiv = ArxivAPIWrapper()
docs = arxiv.run("1605.08386")
docs


# Now, we want to get information about one author, `Caprice Stanley`.
#
# This query returns information about three articles. By default, the query returns information only about three top articles.

# In[6]:


docs = arxiv.run("Caprice Stanley")
docs


# Now, we are trying to find information about non-existing article. In this case, the response is "No good Arxiv Result was found"

# In[7]:


docs = arxiv.run("1605.08386WWW")
docs
