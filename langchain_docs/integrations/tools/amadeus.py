#!/usr/bin/env python
# coding: utf-8

# # Amadeus Toolkit
# 
# This notebook walks you through connecting LangChain to the `Amadeus` travel APIs.
# 
# This `Amadeus` toolkit allows agents to make decision when it comes to travel, especially searching and booking trips with flights.
# 
# To use this toolkit, you will need to have your Amadeus API keys ready, explained in the [Get started Amadeus Self-Service APIs](https://developers.amadeus.com/get-started/get-started-with-self-service-apis-335). Once you've received a AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET, you can input them as environmental variables below.
# 
# Note: Amadeus Self-Service APIs offers a test environment with [free limited data](https://amadeus4dev.github.io/developer-guides/test-data/). This allows developers to build and test their applications before deploying them to production. To access real-time data, you will need to [move to the production environment](https://amadeus4dev.github.io/developer-guides/API-Keys/moving-to-production/).

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  amadeus > /dev/null')


# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-community')


# ## Assign Environmental Variables
# 
# The toolkit will read the AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET environmental variables to authenticate the user, so you need to set them here. 

# In[1]:


# Set environmental variables here
import os

os.environ["AMADEUS_CLIENT_ID"] = "CLIENT_ID"
os.environ["AMADEUS_CLIENT_SECRET"] = "CLIENT_SECRET"

# os.environ["AMADEUS_HOSTNAME"] = "production" or "test"


# ## Create the Amadeus Toolkit and Get Tools
# 
# To start, you need to create the toolkit, so you can access its tools later.

# By default, `AmadeusToolkit` uses `ChatOpenAI` to identify airports closest to a given location. To use it, just set `OPENAI_API_KEY`.
# 

# In[3]:


os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"


# In[4]:


from langchain_community.agent_toolkits.amadeus.toolkit import AmadeusToolkit

toolkit = AmadeusToolkit()
tools = toolkit.get_tools()


# Alternatively, you can use any LLM supported by langchain, e.g. `HuggingFaceHub`. 

# In[ ]:


from langchain_community.llms import HuggingFaceHub

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_HF_API_TOKEN"

llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    model_kwargs={"temperature": 0.5, "max_length": 64},
)

toolkit_hf = AmadeusToolkit(llm=llm)


# ## Use Amadeus Toolkit within an Agent

# In[5]:


from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.tools.render import render_text_description_and_args
from langchain_openai import ChatOpenAI


# In[6]:


llm = ChatOpenAI(temperature=0)

prompt = hub.pull("hwchase17/react-json")
agent = create_react_agent(
    llm,
    tools,
    prompt,
    tools_renderer=render_text_description_and_args,
    output_parser=ReActJsonSingleInputOutputParser(),
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)


# In[8]:


agent_executor.invoke({"input": "What is the name of the airport in Cali, Colombia?"})


# In[7]:


agent_executor.invoke(
    {
        "input": "What is the departure time of the cheapest flight on March 10, 2024 leaving Dallas, Texas before noon to Lincoln, Nebraska?"
    }
)


# In[19]:


agent_executor.invoke(
    {
        "input": "At what time does earliest flight on March 10, 2024 leaving Dallas, Texas to Lincoln, Nebraska land in Nebraska?"
    }
)


# In[8]:


# to execute api correctly, change the querying date to feature
agent_executor.invoke(
    {
        "input": "What is the full travel time for the cheapest flight between Portland, Oregon to Dallas, TX on March 10, 2024?"
    }
)


# In[11]:


agent_executor.invoke(
    {
        "input": "Please draft a concise email from Santiago to Paul, Santiago's travel agent, asking him to book the earliest flight from DFW to DCA on March 10, 2024. Include all flight details in the email."
    }
)


# In[ ]:




