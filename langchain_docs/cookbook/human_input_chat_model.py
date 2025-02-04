#!/usr/bin/env python
# coding: utf-8

# # Human input chat model
# 
# Along with HumanInputLLM, LangChain also provides a pseudo chat model class that can be used for testing, debugging, or educational purposes. This allows you to mock out calls to the chat model and simulate how a human would respond if they received the messages.
# 
# In this notebook, we go over how to use this.
# 
# We start this with using the HumanInputChatModel in an agent.

# In[1]:


from langchain_community.chat_models.human import HumanInputChatModel


# Since we will use the `WikipediaQueryRun` tool in this notebook, you might need to install the `wikipedia` package if you haven't done so already.

# In[2]:


get_ipython().run_line_magic('pip', 'install wikipedia')


# In[3]:


from langchain.agents import AgentType, initialize_agent, load_tools


# In[4]:


tools = load_tools(["wikipedia"])
llm = HumanInputChatModel()


# In[5]:


agent = initialize_agent(
    tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


# In[6]:


agent("What is Bocchi the Rock?")

