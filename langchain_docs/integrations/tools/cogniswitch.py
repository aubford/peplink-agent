#!/usr/bin/env python
# coding: utf-8

# # Cogniswitch Toolkit
#
# CogniSwitch is used to build production ready applications that can consume, organize and retrieve knowledge flawlessly. Using the framework of your choice, in this case Langchain, CogniSwitch helps alleviate the stress of decision making when it comes to, choosing the right storage and retrieval formats. It also eradicates reliability issues and hallucinations when it comes to responses that are generated.
#
# ## Setup
#
# Visit [this page](https://www.cogniswitch.ai/developer?utm_source=langchain&utm_medium=langchainbuild&utm_id=dev) to register a Cogniswitch account.
#
# - Signup with your email and verify your registration
#
# - You will get a mail with a platform token and oauth token for using the services.
#

# In[ ]:


get_ipython().run_line_magic("pip", "install -qU langchain-community")


# ## Import necessary libraries

# In[1]:


import warnings

warnings.filterwarnings("ignore")

import os

from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain_community.agent_toolkits import CogniswitchToolkit
from langchain_openai import ChatOpenAI


# ## Cogniswitch platform token, OAuth token and OpenAI API key

# In[2]:


cs_token = "Your CogniSwitch token"
OAI_token = "Your OpenAI API token"
oauth_token = "Your CogniSwitch authentication token"

os.environ["OPENAI_API_KEY"] = OAI_token


# ## Instantiate the cogniswitch toolkit with the credentials

# In[3]:


cogniswitch_toolkit = CogniswitchToolkit(
    cs_token=cs_token, OAI_token=OAI_token, apiKey=oauth_token
)


# ### Get the list of cogniswitch tools

# In[4]:


tool_lst = cogniswitch_toolkit.get_tools()


# ## Instantiate the LLM

# In[6]:


llm = ChatOpenAI(
    temperature=0,
    openai_api_key=OAI_token,
    max_tokens=1500,
    model_name="gpt-3.5-turbo-0613",
)


# ## Use the LLM with the Toolkit
#
# ### Create an agent with the LLM and Toolkit

# In[7]:


agent_executor = create_conversational_retrieval_agent(llm, tool_lst, verbose=False)


# ### Invoke the agent to upload a URL

# In[9]:


response = agent_executor.invoke("upload this url https://cogniswitch.ai/developer")

print(response["output"])


# ### Invoke the agent to upload a File

# In[10]:


response = agent_executor.invoke("upload this file example_file.txt")

print(response["output"])


# ### Invoke the agent to get the status of a document

# In[11]:


response = agent_executor.invoke("Tell me the status of this document example_file.txt")

print(response["output"])


# ### Invoke the agent with query and get the answer

# In[12]:


response = agent_executor.invoke("How can cogniswitch help develop GenAI applications?")

print(response["output"])
