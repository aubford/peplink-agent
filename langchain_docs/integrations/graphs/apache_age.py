#!/usr/bin/env python
# coding: utf-8

# # Apache AGE
# 
# >[Apache AGE](https://age.apache.org/) is a PostgreSQL extension that provides graph database functionality. AGE is an acronym for A Graph Extension, and is inspired by Bitnine’s fork of PostgreSQL 10, AgensGraph, which is a multi-model database. The goal of the project is to create single storage that can handle both relational and graph model data so that users can use standard ANSI SQL along with openCypher, the Graph query language. The data elements `Apache AGE` stores are nodes, edges connecting them, and attributes of nodes and edges.
# 
# >This notebook shows how to use LLMs to provide a natural language interface to a graph database you can query with the `Cypher` query language.
# 
# >[Cypher](https://en.wikipedia.org/wiki/Cypher_(query_language)) is a declarative graph query language that allows for expressive and efficient data querying in a property graph.
# 

# ## Setting up
# 
# You will need to have a running `Postgre` instance with the AGE extension installed. One option for testing is to run a docker container using the official AGE docker image.
# You can run a local docker container by running the executing the following script:
# 
# ```
# docker run \
#     --name age  \
#     -p 5432:5432 \
#     -e POSTGRES_USER=postgresUser \
#     -e POSTGRES_PASSWORD=postgresPW \
#     -e POSTGRES_DB=postgresDB \
#     -d \
#     apache/age
# ```
# 
# Additional instructions on running in docker can be found [here](https://hub.docker.com/r/apache/age).

# In[1]:


from langchain_community.graphs.age_graph import AGEGraph
from langchain_neo4j import GraphCypherQAChain
from langchain_openai import ChatOpenAI


# In[2]:


conf = {
    "database": "postgresDB",
    "user": "postgresUser",
    "password": "postgresPW",
    "host": "localhost",
    "port": 5432,
}

graph = AGEGraph(graph_name="age_test", conf=conf)


# ## Seeding the database
# 
# Assuming your database is empty, you can populate it using Cypher query language. The following Cypher statement is idempotent, which means the database information will be the same if you run it one or multiple times.

# In[3]:


graph.query(
    """
MERGE (m:Movie {name:"Top Gun"})
WITH m
UNWIND ["Tom Cruise", "Val Kilmer", "Anthony Edwards", "Meg Ryan"] AS actor
MERGE (a:Actor {name:actor})
MERGE (a)-[:ACTED_IN]->(m)
"""
)


# ## Refresh graph schema information
# If the schema of database changes, you can refresh the schema information needed to generate Cypher statements.

# In[4]:


graph.refresh_schema()


# In[5]:


print(graph.schema)


# ## Querying the graph
# 
# We can now use the graph cypher QA chain to ask question of the graph

# In[6]:


chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True, allow_dangerous_requests=True
)


# In[7]:


chain.invoke("Who played in Top Gun?")


# ## Limit the number of results
# You can limit the number of results from the Cypher QA Chain using the `top_k` parameter.
# The default is 10.

# In[8]:


chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0),
    graph=graph,
    verbose=True,
    top_k=2,
    allow_dangerous_requests=True,
)


# In[9]:


chain.invoke("Who played in Top Gun?")


# ## Return intermediate results
# You can return intermediate steps from the Cypher QA Chain using the `return_intermediate_steps` parameter

# In[22]:


chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0),
    graph=graph,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True,
)


# In[23]:


result = chain("Who played in Top Gun?")
print(f"Intermediate steps: {result['intermediate_steps']}")
print(f"Final answer: {result['result']}")


# ## Return direct results
# You can return direct results from the Cypher QA Chain using the `return_direct` parameter

# In[12]:


chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0),
    graph=graph,
    verbose=True,
    return_direct=True,
    allow_dangerous_requests=True,
)


# In[13]:


chain.invoke("Who played in Top Gun?")


# ## Add examples in the Cypher generation prompt
# You can define the Cypher statement you want the LLM to generate for particular questions

# In[14]:


from langchain_core.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher statements for particular questions:
# How many people played in Top Gun?
MATCH (m:Movie {{title:"Top Gun"}})<-[:ACTED_IN]-()
RETURN count(*) AS numberOfActors

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0),
    graph=graph,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    allow_dangerous_requests=True,
)


# In[15]:


chain.invoke("How many people played in Top Gun?")


# ## Use separate LLMs for Cypher and answer generation
# You can use the `cypher_llm` and `qa_llm` parameters to define different llms

# In[16]:


chain = GraphCypherQAChain.from_llm(
    graph=graph,
    cypher_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    qa_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
    verbose=True,
    allow_dangerous_requests=True,
)


# In[17]:


chain.invoke("Who played in Top Gun?")


# ## Ignore specified node and relationship types
# 
# You can use `include_types` or `exclude_types` to ignore parts of the graph schema when generating Cypher statements.

# In[18]:


chain = GraphCypherQAChain.from_llm(
    graph=graph,
    cypher_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    qa_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
    verbose=True,
    exclude_types=["Movie"],
    allow_dangerous_requests=True,
)


# In[19]:


# Inspect graph schema
print(chain.graph_schema)


# ## Validate generated Cypher statements
# You can use the `validate_cypher` parameter to validate and correct relationship directions in generated Cypher statements

# In[20]:


chain = GraphCypherQAChain.from_llm(
    llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    graph=graph,
    verbose=True,
    validate_cypher=True,
    allow_dangerous_requests=True,
)


# In[21]:


chain.invoke("Who played in Top Gun?")

