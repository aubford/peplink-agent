#!/usr/bin/env python
# coding: utf-8

# # Human input LLM
#
# Similar to the fake LLM, LangChain provides a pseudo LLM class that can be used for testing, debugging, or educational purposes. This allows you to mock out calls to the LLM and simulate how a human would respond if they received the prompts.
#
# In this notebook, we go over how to use this.
#
# We start this with using the HumanInputLLM in an agent.

# In[1]:


from langchain_community.llms.human import HumanInputLLM


# In[2]:


from langchain.agents import AgentType, initialize_agent, load_tools


# Since we will use the `WikipediaQueryRun` tool in this notebook, you might need to install the `wikipedia` package if you haven't done so already.

# In[ ]:


get_ipython().run_line_magic("pip", "install wikipedia")


# In[4]:


tools = load_tools(["wikipedia"])
llm = HumanInputLLM(
    prompt_func=lambda prompt: print(
        f"\n===PROMPT====\n{prompt}\n=====END OF PROMPT======"
    )
)


# In[5]:


agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


# In[6]:


agent.run("What is 'Bocchi the Rock!'?")
