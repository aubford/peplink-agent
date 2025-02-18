#!/usr/bin/env python
# coding: utf-8

# # OpenAPI Toolkit
#
# We can construct agents to consume arbitrary APIs, here APIs conformant to the `OpenAPI`/`Swagger` specification.

# In[ ]:


# NOTE: In this example. We must set `allow_dangerous_request=True` to enable the OpenAPI Agent to automatically use the Request Tool.
# This can be dangerous for calling unwanted requests. Please make sure your custom OpenAPI spec (yaml) is safe.
ALLOW_DANGEROUS_REQUEST = True


# ## 1st example: hierarchical planning agent
#
# In this example, we'll consider an approach called hierarchical planning, common in robotics and appearing in recent works for LLMs X robotics. We'll see it's a viable approach to start working with a massive API spec AND to assist with user queries that require multiple steps against the API.
#
# The idea is simple: to get coherent agent behavior over long sequences behavior & to save on tokens, we'll separate concerns: a "planner" will be responsible for what endpoints to call and a "controller" will be responsible for how to call them.
#
# In the initial implementation, the planner is an LLM chain that has the name and a short description for each endpoint in context. The controller is an LLM agent that is instantiated with documentation for only the endpoints for a particular plan. There's a lot left to get this working very robustly :)
#
# ---

# ### To start, let's collect some OpenAPI specs.

# In[1]:


import os

import yaml


# You will be able to get OpenAPI specs from here: [APIs-guru/openapi-directory](https://github.com/APIs-guru/openapi-directory)

# In[2]:


get_ipython().system(
    "wget https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml -O openai_openapi.yaml"
)
get_ipython().system(
    "wget https://www.klarna.com/us/shopping/public/openai/v0/api-docs -O klarna_openapi.yaml"
)
get_ipython().system(
    "wget https://raw.githubusercontent.com/APIs-guru/openapi-directory/main/APIs/spotify.com/1.0.0/openapi.yaml -O spotify_openapi.yaml"
)


# In[3]:


from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec


# In[4]:


with open("openai_openapi.yaml") as f:
    raw_openai_api_spec = yaml.load(f, Loader=yaml.Loader)
openai_api_spec = reduce_openapi_spec(raw_openai_api_spec)

with open("klarna_openapi.yaml") as f:
    raw_klarna_api_spec = yaml.load(f, Loader=yaml.Loader)
klarna_api_spec = reduce_openapi_spec(raw_klarna_api_spec)

with open("spotify_openapi.yaml") as f:
    raw_spotify_api_spec = yaml.load(f, Loader=yaml.Loader)
spotify_api_spec = reduce_openapi_spec(raw_spotify_api_spec)


# ---
#
# We'll work with the Spotify API as one of the examples of a somewhat complex API. There's a bit of auth-related setup to do if you want to replicate this.
#
# - You'll have to set up an application in the Spotify developer console, documented [here](https://developer.spotify.com/documentation/general/guides/authorization/), to get credentials: `CLIENT_ID`, `CLIENT_SECRET`, and `REDIRECT_URI`.
# - To get an access tokens (and keep them fresh), you can implement the oauth flows, or you can use `spotipy`. If you've set your Spotify creedentials as environment variables `SPOTIPY_CLIENT_ID`, `SPOTIPY_CLIENT_SECRET`, and `SPOTIPY_REDIRECT_URI`, you can use the helper functions below:

# In[5]:


import spotipy.util as util
from langchain_community.utilities.requests import RequestsWrapper


def construct_spotify_auth_headers(raw_spec: dict):
    scopes = list(
        raw_spec["components"]["securitySchemes"]["oauth_2_0"]["flows"][
            "authorizationCode"
        ]["scopes"].keys()
    )
    access_token = util.prompt_for_user_token(scope=",".join(scopes))
    return {"Authorization": f"Bearer {access_token}"}


# Get API credentials.
headers = construct_spotify_auth_headers(raw_spotify_api_spec)
requests_wrapper = RequestsWrapper(headers=headers)


# ### How big is this spec?

# In[6]:


endpoints = [
    (route, operation)
    for route, operations in raw_spotify_api_spec["paths"].items()
    for operation in operations
    if operation in ["get", "post"]
]
len(endpoints)


# In[7]:


import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")


def count_tokens(s):
    return len(enc.encode(s))


count_tokens(yaml.dump(raw_spotify_api_spec))


# ### Let's see some examples!
#
# Starting with GPT-4. (Some robustness iterations under way for GPT-3 family.)

# In[8]:


from langchain_community.agent_toolkits.openapi import planner
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)


# In[9]:


# NOTE: set allow_dangerous_requests manually for security concern https://python.langchain.com/docs/security
spotify_agent = planner.create_openapi_agent(
    spotify_api_spec,
    requests_wrapper,
    llm,
    allow_dangerous_requests=ALLOW_DANGEROUS_REQUEST,
)
user_query = (
    "make me a playlist with the first song from kind of blue. call it machine blues."
)
spotify_agent.invoke(user_query)


# In[12]:


user_query = "give me a song I'd like, make it blues-ey"
spotify_agent.invoke(user_query)


# #### Try another API.
#

# In[23]:


headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
openai_requests_wrapper = RequestsWrapper(headers=headers)


# In[28]:


# Meta!
llm = ChatOpenAI(model_name="gpt-4", temperature=0.25)
openai_agent = planner.create_openapi_agent(
    openai_api_spec, openai_requests_wrapper, llm
)
user_query = "generate a short piece of advice"
openai_agent.invoke(user_query)


# Takes awhile to get there!

# ## 2nd example: "json explorer" agent
#
# Here's an agent that's not particularly practical, but neat! The agent has access to 2 toolkits. One comprises tools to interact with json: one tool to list the keys of a json object and another tool to get the value for a given key. The other toolkit comprises `requests` wrappers to send GET and POST requests. This agent consumes a lot calls to the language model, but does a surprisingly decent job.
#

# In[29]:


from langchain_community.agent_toolkits import OpenAPIToolkit, create_openapi_agent
from langchain_community.tools.json.tool import JsonSpec
from langchain_openai import OpenAI


# In[32]:


with open("openai_openapi.yaml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
json_spec = JsonSpec(dict_=data, max_value_length=4000)


openapi_toolkit = OpenAPIToolkit.from_llm(
    OpenAI(temperature=0), json_spec, openai_requests_wrapper, verbose=True
)
openapi_agent_executor = create_openapi_agent(
    llm=OpenAI(temperature=0),
    toolkit=openapi_toolkit,
    allow_dangerous_requests=ALLOW_DANGEROUS_REQUEST,
    verbose=True,
)


# In[33]:


openapi_agent_executor.run(
    "Make a post request to openai /completions. The prompt should be 'tell me a joke.'"
)
