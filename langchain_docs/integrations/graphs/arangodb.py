#!/usr/bin/env python
# coding: utf-8

# # ArangoDB
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arangodb/interactive_tutorials/blob/master/notebooks/Langchain.ipynb)
#
# >[ArangoDB](https://github.com/arangodb/arangodb) is a scalable graph database system to drive value from
# >connected data, faster. Native graphs, an integrated search engine, and JSON support, via
# >a single query language. `ArangoDB` runs on-prem or in the cloud.
#
# This notebook shows how to use LLMs to provide a natural language interface to an [ArangoDB](https://github.com/arangodb/arangodb#readme) database.

# ## Setting up
#
# You can get a local `ArangoDB` instance running via the [ArangoDB Docker image](https://hub.docker.com/_/arangodb):
#
# ```
# docker run -p 8529:8529 -e ARANGO_ROOT_PASSWORD= arangodb/arangodb
# ```
#
# An alternative is to use the [ArangoDB Cloud Connector package](https://github.com/arangodb/adb-cloud-connector#readme) to get a temporary cloud instance running:

# In[1]:


get_ipython().run_cell_magic(
    "capture",
    "",
    "%pip install --upgrade --quiet  python-arango # The ArangoDB Python Driver\n%pip install --upgrade --quiet  adb-cloud-connector # The ArangoDB Cloud Instance provisioner\n%pip install --upgrade --quiet  langchain-openai\n%pip install --upgrade --quiet  langchain\n",
)


# In[2]:


# Instantiate ArangoDB Database
import json

from adb_cloud_connector import get_temp_credentials
from arango import ArangoClient

con = get_temp_credentials()

db = ArangoClient(hosts=con["url"]).db(
    con["dbName"], con["username"], con["password"], verify=True
)

print(json.dumps(con, indent=2))


# In[3]:


# Instantiate the ArangoDB-LangChain Graph
from langchain_community.graphs import ArangoGraph

graph = ArangoGraph(db)


# ## Populating database
#
# We will rely on the `Python Driver` to import our [GameOfThrones](https://github.com/arangodb/example-datasets/tree/master/GameOfThrones) data into our database.

# In[4]:


if db.has_graph("GameOfThrones"):
    db.delete_graph("GameOfThrones", drop_collections=True)

db.create_graph(
    "GameOfThrones",
    edge_definitions=[
        {
            "edge_collection": "ChildOf",
            "from_vertex_collections": ["Characters"],
            "to_vertex_collections": ["Characters"],
        },
    ],
)

documents = [
    {
        "_key": "NedStark",
        "name": "Ned",
        "surname": "Stark",
        "alive": True,
        "age": 41,
        "gender": "male",
    },
    {
        "_key": "CatelynStark",
        "name": "Catelyn",
        "surname": "Stark",
        "alive": False,
        "age": 40,
        "gender": "female",
    },
    {
        "_key": "AryaStark",
        "name": "Arya",
        "surname": "Stark",
        "alive": True,
        "age": 11,
        "gender": "female",
    },
    {
        "_key": "BranStark",
        "name": "Bran",
        "surname": "Stark",
        "alive": True,
        "age": 10,
        "gender": "male",
    },
]

edges = [
    {"_to": "Characters/NedStark", "_from": "Characters/AryaStark"},
    {"_to": "Characters/NedStark", "_from": "Characters/BranStark"},
    {"_to": "Characters/CatelynStark", "_from": "Characters/AryaStark"},
    {"_to": "Characters/CatelynStark", "_from": "Characters/BranStark"},
]

db.collection("Characters").import_bulk(documents)
db.collection("ChildOf").import_bulk(edges)


# ## Getting and setting the ArangoDB schema
#
# An initial `ArangoDB Schema` is generated upon instantiating the `ArangoDBGraph` object. Below are the schema's getter & setter methods should you be interested in viewing or modifying the schema:

# In[5]:


# The schema should be empty here,
# since `graph` was initialized prior to ArangoDB Data ingestion (see above).

import json

print(json.dumps(graph.schema, indent=4))


# In[6]:


graph.set_schema()


# In[7]:


# We can now view the generated schema

import json

print(json.dumps(graph.schema, indent=4))


# ## Querying the ArangoDB database
#
# We can now use the `ArangoDB Graph` QA Chain to inquire about our data

# In[ ]:


import os

os.environ["OPENAI_API_KEY"] = "your-key-here"


# In[9]:


from langchain.chains import ArangoGraphQAChain
from langchain_openai import ChatOpenAI

chain = ArangoGraphQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True
)


# In[10]:


chain.run("Is Ned Stark alive?")


# In[11]:


chain.run("How old is Arya Stark?")


# In[12]:


chain.run("Are Arya Stark and Ned Stark related?")


# In[13]:


chain.run("Does Arya Stark have a dead parent?")


# ## Chain modifiers

# You can alter the values of the following `ArangoDBGraphQAChain` class variables to modify the behaviour of your chain results
#

# In[14]:


# Specify the maximum number of AQL Query Results to return
chain.top_k = 10

# Specify whether or not to return the AQL Query in the output dictionary
chain.return_aql_query = True

# Specify whether or not to return the AQL JSON Result in the output dictionary
chain.return_aql_result = True

# Specify the maximum amount of AQL Generation attempts that should be made
chain.max_aql_generation_attempts = 5

# Specify a set of AQL Query Examples, which are passed to
# the AQL Generation Prompt Template to promote few-shot-learning.
# Defaults to an empty string.
chain.aql_examples = """
# Is Ned Stark alive?
RETURN DOCUMENT('Characters/NedStark').alive

# Is Arya Stark the child of Ned Stark?
FOR e IN ChildOf
    FILTER e._from == "Characters/AryaStark" AND e._to == "Characters/NedStark"
    RETURN e
"""


# In[15]:


chain.run("Is Ned Stark alive?")

# chain("Is Ned Stark alive?") # Returns a dictionary with the AQL Query & AQL Result


# In[16]:


chain.run("Is Bran Stark the child of Ned Stark?")
