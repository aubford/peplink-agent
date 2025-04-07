#!/usr/bin/env python
# coding: utf-8

# # Prolog
# 
# LangChain tools that use Prolog rules to generate answers.
# 
# ## Overview
# 
# The PrologTool class allows the generation of langchain tools that use Prolog rules to generate answers.
# 
# ## Setup
# 
# Let's use the following Prolog rules in the file family.pl:
# 
# parent(john, bianca, mary).\
# parent(john, bianca, michael).\
# parent(peter, patricia, jennifer).\
# partner(X, Y) :- parent(X, Y, _).

# In[1]:


#!pip install langchain-prolog

from langchain_prolog import PrologConfig, PrologRunnable, PrologTool

TEST_SCRIPT = "family.pl"


# ## Instantiation
# 
# First create the Prolog tool:

# In[2]:


schema = PrologRunnable.create_schema("parent", ["men", "women", "child"])
config = PrologConfig(
    rules_file=TEST_SCRIPT,
    query_schema=schema,
)
prolog_tool = PrologTool(
    prolog_config=config,
    name="family_query",
    description="""
        Query family relationships using Prolog.
        parent(X, Y, Z) implies only that Z is a child of X and Y.
        Input can be a query string like 'parent(john, X, Y)' or 'john, X, Y'"
        You have to specify 3 parameters: men, woman, child. Do not use quotes.
    """,
)


# ## Invocation
# 
# ### Using a Prolog tool with an LLM and function calling

# In[3]:


#!pip install python-dotenv

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

#!pip install langchain-openai

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


# To use the tool, bind it to the LLM model:

# In[4]:


llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools([prolog_tool])


# and then query the model:

# In[5]:


query = "Who are John's children?"
messages = [HumanMessage(query)]
response = llm_with_tools.invoke(messages)


# The LLM will respond with a tool call request:

# In[6]:


messages.append(response)
response.tool_calls[0]


# The tool takes this request and queries the Prolog database:

# In[7]:


tool_msg = prolog_tool.invoke(response.tool_calls[0])


# The tool returns a list with all the solutions for the query:

# In[8]:


messages.append(tool_msg)
tool_msg


# That we then pass to the LLM, and the LLM answers the original query using the tool response:

# In[9]:


answer = llm_with_tools.invoke(messages)
print(answer.content)


# ## Chaining
# 
# ### Using a Prolog Tool with an agent

# To use the prolog tool with an agent, pass it to the agent's constructor:

# In[10]:


#!pip install langgraph

from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(llm, [prolog_tool])


# The agent takes the query and use the Prolog tool if needed:

# In[11]:


messages = agent_executor.invoke({"messages": [("human", query)]})


# Then the agent receives​ the tool response and generates the answer:

# In[12]:


messages["messages"][-1].pretty_print()


# ## API reference
# 
# See https://langchain-prolog.readthedocs.io/en/latest/modules.html for detail.

# In[ ]:




