#!/usr/bin/env python
# coding: utf-8

# # MultiOn Toolkit
#  
# [MultiON](https://www.multion.ai/blog/multion-building-a-brighter-future-for-humanity-with-ai-agents) has built an AI Agent that can interact with a broad array of web services and applications. 
# 
# This notebook walks you through connecting LangChain to the `MultiOn` Client in your browser. 
# 
# This enables custom agentic workflow that utilize the power of MultiON agents.
#  
# To use this toolkit, you will need to add `MultiOn Extension` to your browser: 
# 
# * Create a [MultiON account](https://app.multion.ai/login?callbackUrl=%2Fprofile). 
# * Add  [MultiOn extension for Chrome](https://multion.notion.site/Download-MultiOn-ddddcfe719f94ab182107ca2612c07a5).

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  multion langchain -q')


# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-community')


# In[37]:


from langchain_community.agent_toolkits import MultionToolkit

toolkit = MultionToolkit()
toolkit


# In[38]:


tools = toolkit.get_tools()
tools


# ## MultiOn Setup
# 
# Once you have created an account, create an API key at https://app.multion.ai/. 
# 
# Login to establish connection with your extension.

# In[39]:


# Authorize connection to your Browser extention
import multion

multion.login()


# ## Use Multion Toolkit within an Agent
# 
# This will use MultiON chrome extension to perform the desired actions.
# 
# We can run the below, and view the [trace](https://smith.langchain.com/public/34aaf36d-204a-4ce3-a54e-4a0976f09670/r) to see:
# 
# * The agent uses the `create_multion_session` tool
# * It then uses MultiON to execute the query

# In[40]:


from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI


# In[41]:


# Prompt
instructions = """You are an assistant."""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)


# In[32]:


# LLM
llm = ChatOpenAI(temperature=0)


# In[42]:


# Agent
agent = create_openai_functions_agent(llm, toolkit.get_tools(), prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=False,
)


# In[46]:


agent_executor.invoke(
    {
        "input": "Use multion to explain how AlphaCodium works, a recently released code language model."
    }
)

