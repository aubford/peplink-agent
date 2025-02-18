#!/usr/bin/env python
# coding: utf-8

# # Weather
#
# >[OpenWeatherMap](https://openweathermap.org/) is an open-source weather service provider
#
# This loader fetches the weather data from the OpenWeatherMap's OneCall API, using the pyowm Python package. You must initialize the loader with your OpenWeatherMap API token and the names of the cities you want the weather data for.

# In[ ]:


from langchain_community.document_loaders import WeatherDataLoader


# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  pyowm")


# In[ ]:


# Set API key either by passing it in to constructor directly
# or by setting the environment variable "OPENWEATHERMAP_API_KEY".

from getpass import getpass

OPENWEATHERMAP_API_KEY = getpass()


# In[ ]:


loader = WeatherDataLoader.from_params(
    ["chennai", "vellore"], openweathermap_api_key=OPENWEATHERMAP_API_KEY
)


# In[ ]:


documents = loader.load()
documents
