#!/usr/bin/env python
# coding: utf-8

# # JSON Toolkit
# 
# This notebook showcases an agent interacting with large `JSON/dict` objects. 
# This is useful when you want to answer questions about a JSON blob that's too large to fit in the context window of an LLM. The agent is able to iteratively explore the blob to find what it needs to answer the user's question.
# 
# In the below example, we are using the OpenAPI spec for the OpenAI API, which you can find [here](https://github.com/openai/openai-openapi/blob/master/openapi.yaml).
# 
# We will use the JSON agent to answer some questions about the API spec.

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-community')


# ## Initialization

# In[2]:


import yaml
from langchain_community.agent_toolkits import JsonToolkit, create_json_agent
from langchain_community.tools.json.tool import JsonSpec
from langchain_openai import OpenAI


# In[4]:


with open("openai_openapi.yml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
json_spec = JsonSpec(dict_={}, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)

json_agent_executor = create_json_agent(
    llm=OpenAI(temperature=0), toolkit=json_toolkit, verbose=True
)


# ## Individual tools
# 
# Let's see what individual tools are inside the Jira toolkit.

# In[5]:


[(el.name, el.description) for el in json_toolkit.get_tools()]


# ## Example: getting the required POST parameters for a request

# In[5]:


json_agent_executor.run(
    "What are the required parameters in the request body to the /completions endpoint?"
)


# In[ ]:




