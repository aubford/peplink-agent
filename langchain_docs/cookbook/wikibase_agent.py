#!/usr/bin/env python
# coding: utf-8

# # Wikibase Agent
# 
# This notebook demonstrates a very simple wikibase agent that uses sparql generation. Although this code is intended to work against any
# wikibase instance, we use http://wikidata.org for testing.
# 
# If you are interested in wikibases and sparql, please consider helping to improve this agent. Look [here](https://github.com/donaldziff/langchain-wikibase) for more details and open questions.
# 

# ## Preliminaries

# ### API keys and other secrets
# 
# We use an `.ini` file, like this: 
# ```
# [OPENAI]
# OPENAI_API_KEY=xyzzy
# [WIKIDATA]
# WIKIDATA_USER_AGENT_HEADER=argle-bargle
# ```

# In[1]:


import configparser

config = configparser.ConfigParser()
config.read("./secrets.ini")


# ### OpenAI API Key
# 
# An OpenAI API key is required unless you modify the code below to use another LLM provider.

# In[2]:


openai_api_key = config["OPENAI"]["OPENAI_API_KEY"]
import os

os.environ.update({"OPENAI_API_KEY": openai_api_key})


# ### Wikidata user-agent header
# 
# Wikidata policy requires a user-agent header. See https://meta.wikimedia.org/wiki/User-Agent_policy. However, at present this policy is not strictly enforced.

# In[3]:


wikidata_user_agent_header = (
    None
    if not config.has_section("WIKIDATA")
    else config["WIKIDATA"]["WIKIDATA_USER_AGENT_HEADER"]
)


# ### Enable tracing if desired

# In[4]:


# import os
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_PROJECT"] = "default" # Make sure this session actually exists.


# # Tools
# 
# Three tools are provided for this simple agent:
# * `ItemLookup`: for finding the q-number of an item
# * `PropertyLookup`: for finding the p-number of a property
# * `SparqlQueryRunner`: for running a sparql query

# ## Item and Property lookup
# 
# Item and Property lookup are implemented in a single method, using an elastic search endpoint. Not all wikibase instances have it, but wikidata does, and that's where we'll start.

# In[5]:


def get_nested_value(o: dict, path: list) -> any:
    current = o
    for key in path:
        try:
            current = current[key]
        except KeyError:
            return None
    return current


from typing import Optional

import requests


def vocab_lookup(
    search: str,
    entity_type: str = "item",
    url: str = "https://www.wikidata.org/w/api.php",
    user_agent_header: str = wikidata_user_agent_header,
    srqiprofile: str = None,
) -> Optional[str]:
    headers = {"Accept": "application/json"}
    if wikidata_user_agent_header is not None:
        headers["User-Agent"] = wikidata_user_agent_header

    if entity_type == "item":
        srnamespace = 0
        srqiprofile = "classic_noboostlinks" if srqiprofile is None else srqiprofile
    elif entity_type == "property":
        srnamespace = 120
        srqiprofile = "classic" if srqiprofile is None else srqiprofile
    else:
        raise ValueError("entity_type must be either 'property' or 'item'")

    params = {
        "action": "query",
        "list": "search",
        "srsearch": search,
        "srnamespace": srnamespace,
        "srlimit": 1,
        "srqiprofile": srqiprofile,
        "srwhat": "text",
        "format": "json",
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        title = get_nested_value(response.json(), ["query", "search", 0, "title"])
        if title is None:
            return f"I couldn't find any {entity_type} for '{search}'. Please rephrase your request and try again"
        # if there is a prefix, strip it off
        return title.split(":")[-1]
    else:
        return "Sorry, I got an error. Please try again."


# In[6]:


print(vocab_lookup("Malin 1"))


# In[7]:


print(vocab_lookup("instance of", entity_type="property"))


# In[8]:


print(vocab_lookup("Ceci n'est pas un q-item"))


# ## Sparql runner 

# This tool runs sparql - by default, wikidata is used.

# In[9]:


import json
from typing import Any, Dict, List

import requests


def run_sparql(
    query: str,
    url="https://query.wikidata.org/sparql",
    user_agent_header: str = wikidata_user_agent_header,
) -> List[Dict[str, Any]]:
    headers = {"Accept": "application/json"}
    if wikidata_user_agent_header is not None:
        headers["User-Agent"] = wikidata_user_agent_header

    response = requests.get(
        url, headers=headers, params={"query": query, "format": "json"}
    )

    if response.status_code != 200:
        return "That query failed. Perhaps you could try a different one?"
    results = get_nested_value(response.json(), ["results", "bindings"])
    return json.dumps(results)


# In[10]:


run_sparql("SELECT (COUNT(?children) as ?count) WHERE { wd:Q1339 wdt:P40 ?children . }")


# # Agent

# ## Wrap the tools

# In[11]:


import re
from typing import List, Union

from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    LLMSingleActionAgent,
    Tool,
)
from langchain.chains import LLMChain
from langchain.prompts import StringPromptTemplate
from langchain_core.agents import AgentAction, AgentFinish


# In[12]:


# Define which tools the agent can use to answer user queries
tools = [
    Tool(
        name="ItemLookup",
        func=(lambda x: vocab_lookup(x, entity_type="item")),
        description="useful for when you need to know the q-number for an item",
    ),
    Tool(
        name="PropertyLookup",
        func=(lambda x: vocab_lookup(x, entity_type="property")),
        description="useful for when you need to know the p-number for a property",
    ),
    Tool(
        name="SparqlQueryRunner",
        func=run_sparql,
        description="useful for getting results from a wikibase",
    ),
]


# ## Prompts

# In[13]:


# Set up the base template
template = """
Answer the following questions by running a sparql query against a wikibase where the p and q items are 
completely unknown to you. You will need to discover the p and q items before you can generate the sparql.
Do not assume you know the p and q items for any concepts. Always use tools to find all p and q items.
After you generate the sparql, you should run it. The results will be returned in json. 
Summarize the json results in natural language.

You may assume the following prefixes:
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>

When generating sparql:
* Try to avoid "count" and "filter" queries if possible
* Never enclose the sparql in back-quotes

You have access to the following tools:

{tools}

Use the following format:

Question: the input question for which you must provide a natural language answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}"""


# In[14]:


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


# In[15]:


prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"],
)


# ## Output parser 
# This is unchanged from langchain docs

# In[16]:


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


# In[17]:


output_parser = CustomOutputParser()


# ## Specify the LLM model

# In[18]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)


# ## Agent and agent executor

# In[19]:


# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)


# In[20]:


tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)


# In[21]:


agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)


# ## Run it!

# In[22]:


# If you prefer in-line tracing, uncomment this line
# agent_executor.agent.llm_chain.verbose = True


# In[23]:


agent_executor.run("How many children did J.S. Bach have?")


# In[24]:


agent_executor.run(
    "What is the Basketball-Reference.com NBA player ID of Hakeem Olajuwon?"
)


# In[ ]:




