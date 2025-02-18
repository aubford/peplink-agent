#!/usr/bin/env python
# coding: utf-8

# # Fake LLM
# LangChain provides a fake LLM class that can be used for testing. This allows you to mock out calls to the LLM and simulate what would happen if the LLM responded in a certain way.
#
# In this notebook we go over how to use this.
#
# We start this with using the FakeLLM in an agent.

# In[1]:


from langchain_community.llms.fake import FakeListLLM


# In[2]:


from langchain.agents import AgentType, initialize_agent, load_tools


# In[3]:


tools = load_tools(["python_repl"])


# In[16]:


responses = ["Action: Python REPL\nAction Input: print(2 + 2)", "Final Answer: 4"]
llm = FakeListLLM(responses=responses)


# In[17]:


agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


# In[18]:


agent.invoke("whats 2 + 2")


# In[ ]:
