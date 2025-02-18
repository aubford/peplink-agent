#!/usr/bin/env python
# coding: utf-8

# # Human as a tool
#
# Human are AGI so they can certainly be used as a tool to help out AI agent
# when it is confused.

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  langchain-community")


# In[1]:


from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import ChatOpenAI, OpenAI

llm = ChatOpenAI(temperature=0.0)
math_llm = OpenAI(temperature=0.0)
tools = load_tools(
    ["human", "llm-math"],
    llm=math_llm,
)

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


# In the above code you can see the tool takes input directly from command line.
# You can customize `prompt_func` and `input_func` according to your need (as shown below).

# In[2]:


agent_chain.run("What's my friend Eric's surname?")
# Answer with 'Zhu'


# ## Configuring the Input Function
#
# By default, the `HumanInputRun` tool uses the python `input` function to get input from the user.
# You can customize the input_func to be anything you'd like.
# For instance, if you want to accept multi-line input, you could do the following:

# In[8]:


def get_input() -> str:
    print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        contents.append(line)
    return "\n".join(contents)


# You can modify the tool when loading
tools = load_tools(["human", "ddg-search"], llm=math_llm, input_func=get_input)


# In[9]:


# Or you can directly instantiate the tool
from langchain_community.tools import HumanInputRun

tool = HumanInputRun(input_func=get_input)


# In[10]:


agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


# In[12]:


agent_chain.run("I need help attributing a quote")


# In[ ]:
