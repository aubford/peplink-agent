#!/usr/bin/env python
# coding: utf-8

# # OpenWeatherMap
# 
# This notebook goes over how to use the `OpenWeatherMap` component to fetch weather information.
# 
# First, you need to sign up for an `OpenWeatherMap API` key:
# 
# 1. Go to OpenWeatherMap and sign up for an API key [here](https://openweathermap.org/api/)
# 2. pip install pyowm
# 
# Then we will need to set some environment variables:
# 1. Save your API KEY into OPENWEATHERMAP_API_KEY env variable
# 
# ## Use the wrapper

# In[9]:


import os

from langchain_community.utilities import OpenWeatherMapAPIWrapper

os.environ["OPENWEATHERMAP_API_KEY"] = ""

weather = OpenWeatherMapAPIWrapper()


# In[10]:


weather_data = weather.run("London,GB")
print(weather_data)


# ## Use the tool

# In[11]:


import os

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import OpenAI

os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENWEATHERMAP_API_KEY"] = ""

llm = OpenAI(temperature=0)

tools = load_tools(["openweathermap-api"], llm)

agent_chain = initialize_agent(
    tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


# In[12]:


agent_chain.run("What's the weather like in London?")

