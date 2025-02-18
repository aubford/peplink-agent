#!/usr/bin/env python
# coding: utf-8

# # Steam Toolkit
#
# >[Steam (Wikipedia)](https://en.wikipedia.org/wiki/Steam_(service)) is a video game digital distribution service and storefront developed by `Valve Corporation`. It provides game updates automatically for Valve's games, and expanded to distributing third-party titles. `Steam` offers various features, like game server matchmaking with Valve Anti-Cheat measures, social networking, and game streaming services.
#
# >[Steam](https://store.steampowered.com/about/) is the ultimate destination for playing, discussing, and creating games.
#
# Steam toolkit has two tools:
# - `Game Details`
# - `Recommended Games`
#
# This notebook provides a walkthrough of using Steam API with LangChain to retrieve Steam game recommendations based on your current Steam Game Inventory or to gather information regarding some Steam Games which you provide.
#
# ## Setting up
#
# We have to install two python libraries.

# ## Imports

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet  python-steam-api python-decouple"
)


# ## Assign Environmental Variables
# To use this toolkit, please have your OpenAI API Key, Steam API key (from [here](https://steamcommunity.com/dev/apikey)) and your own SteamID handy. Once you have received a Steam API Key, you can input it as an environmental variable below.
# The toolkit will read the "STEAM_KEY" API Key as an environmental variable to authenticate you so please set them here. You will also need to set your "OPENAI_API_KEY" and your "STEAM_ID".

# In[6]:


import os

os.environ["STEAM_KEY"] = "xyz"
os.environ["STEAM_ID"] = "123"
os.environ["OPENAI_API_KEY"] = "abc"


# ## Initialization:
# Initialize the LLM, SteamWebAPIWrapper, SteamToolkit and most importantly the langchain agent to process your query!
# ## Example

# In[ ]:


from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.steam.toolkit import SteamToolkit
from langchain_community.utilities.steam import SteamWebAPIWrapper
from langchain_openai import OpenAI


# In[7]:


llm = OpenAI(temperature=0)
Steam = SteamWebAPIWrapper()
toolkit = SteamToolkit.from_steam_api_wrapper(Steam)
agent = initialize_agent(
    toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


# In[8]:


out = agent("can you give the information about the game Terraria")
print(out)
