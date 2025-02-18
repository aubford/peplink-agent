#!/usr/bin/env python
# coding: utf-8

# # Semantic Scholar API Tool
#
# This notebook demos how to use the semantic scholar tool with an agent.

# In[1]:


# start by installing semanticscholar api
get_ipython().run_line_magic("pip", "install --upgrade --quiet  semanticscholar")


# In[2]:


from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI


# In[3]:


instructions = """You are an expert researcher."""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)


# In[4]:


llm = ChatOpenAI(temperature=0)


# In[7]:


from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun

tools = [SemanticScholarQueryRun()]


# In[8]:


agent = create_openai_functions_agent(llm, tools, prompt)


# In[9]:


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)


# In[12]:


agent_executor.invoke(
    {
        "input": "What are some biases in the large language models? How have people tried to mitigate them? "
        "show me a list of papers and techniques. Based on your findings write new research questions "
        "to work on. Break down the task into subtasks for search. Use the search tool"
    }
)


# In[ ]:
