#!/usr/bin/env python
# coding: utf-8

# # AINetwork Toolkit
#
# >[AI Network](https://www.ainetwork.ai/build-on-ain) is a layer 1 blockchain designed to accommodate large-scale AI models, utilizing a decentralized GPU network powered by the [$AIN token](https://www.ainetwork.ai/token), enriching AI-driven `NFTs` (`AINFTs`).
# >
# >The `AINetwork Toolkit` is a set of tools for interacting with the [AINetwork Blockchain](https://www.ainetwork.ai/public/whitepaper.pdf). These tools allow you to transfer `AIN`, read and write values, create apps, and set permissions for specific paths within the blockchain database.

# ## Installing dependencies
#
# Before using the AINetwork Toolkit, you need to install the ain-py package. You can install it with pip:
#

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet  ain-py langchain-community"
)


# ## Set environmental variables
#
# You need to set the `AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY` environmental variable to your AIN Blockchain Account Private Key.

# In[2]:


import os

os.environ["AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY"] = ""


# ### Get AIN Blockchain private key

# In[4]:


import os

from ain.account import Account

if os.environ.get("AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY", None):
    account = Account(os.environ["AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY"])
else:
    account = Account.create()
    os.environ["AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY"] = account.private_key
    print(
        f"""
address: {account.address}
private_key: {account.private_key}
"""
    )
# IMPORTANT: If you plan to use this account in the future, make sure to save the
#  private key in a secure place. Losing access to your private key means losing
#  access to your account.


# ## Initialize the AINetwork Toolkit
#
# You can initialize the AINetwork Toolkit like this:

# In[4]:


from langchain_community.agent_toolkits.ainetwork.toolkit import AINetworkToolkit

toolkit = AINetworkToolkit()
tools = toolkit.get_tools()
address = tools[0].interface.wallet.defaultAccount.address


# ## Initialize the Agent with the AINetwork Toolkit
#
# You can initialize the agent with the AINetwork Toolkit like this:

# In[5]:


from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
)


# ## Example Usage
#
# Here are some examples of how you can use the agent with the AINetwork Toolkit:

# ### Define App name to test

# In[6]:


appName = f"langchain_demo_{address.lower()}"


# ### Create an app in the AINetwork Blockchain database

# In[7]:


print(
    agent.run(
        f"Create an app in the AINetwork Blockchain database with the name {appName}"
    )
)


# ### Set a value at a given path in the AINetwork Blockchain database

# In[8]:


print(
    agent.run(f"Set the value {{1: 2, '34': 56}} at the path /apps/{appName}/object .")
)


# ### Set permissions for a path in the AINetwork Blockchain database

# In[9]:


print(
    agent.run(
        f"Set the write permissions for the path /apps/{appName}/user/$from with the"
        " eval string auth.addr===$from ."
    )
)


# ### Retrieve the permissions for a path in the AINetwork Blockchain database

# In[10]:


print(agent.run(f"Retrieve the permissions for the path /apps/{appName}."))


# ### Get AIN from faucet

# In[11]:


get_ipython().system("curl http://faucet.ainetwork.ai/api/test/{address}/")


# ### Get AIN Balance

# In[12]:


print(agent.run(f"Check AIN balance of {address}"))


# ### Transfer AIN

# In[13]:


print(
    agent.run(
        "Transfer 100 AIN to the address 0x19937b227b1b13f29e7ab18676a89ea3bdea9c5b"
    )
)
