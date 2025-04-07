#!/usr/bin/env python
# coding: utf-8

# # How to better prompt when doing SQL question-answering
# 
# In this guide we'll go over prompting strategies to improve SQL query generation using [create_sql_query_chain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.sql_database.query.create_sql_query_chain.html). We'll largely focus on methods for getting relevant database-specific information in your prompt.
# 
# We will cover: 
# 
# - How the dialect of the LangChain [SQLDatabase](https://python.langchain.com/api_reference/community/utilities/langchain_community.utilities.sql_database.SQLDatabase.html) impacts the prompt of the chain;
# - How to format schema information into the prompt using `SQLDatabase.get_context`;
# - How to build and select [few-shot examples](/docs/concepts/few_shot_prompting/) to assist the model.
# 
# ## Setup
# 
# First, get required packages and set environment variables:

# In[2]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  langchain langchain-community langchain-experimental langchain-openai')


# In[ ]:


# Uncomment the below to use LangSmith. Not required.
# import os
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
# os.environ["LANGSMITH_TRACING"] = "true"


# The below example will use a SQLite connection with Chinook database. Follow [these installation steps](https://database.guide/2-sample-databases-sqlite/) to create `Chinook.db` in the same directory as this notebook:
# 
# * Save [this file](https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql) as `Chinook_Sqlite.sql`
# * Run `sqlite3 Chinook.db`
# * Run `.read Chinook_Sqlite.sql`
# * Test `SELECT * FROM Artist LIMIT 10;`
# 
# Now, `Chinook.db` is in our directory and we can interface with it using the SQLAlchemy-driven `SQLDatabase` class:

# In[1]:


from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db", sample_rows_in_table_info=3)
print(db.dialect)
print(db.get_usable_table_names())
print(db.run("SELECT * FROM Artist LIMIT 10;"))


# ## Dialect-specific prompting
# 
# One of the simplest things we can do is make our prompt specific to the SQL dialect we're using. When using the built-in [create_sql_query_chain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.sql_database.query.create_sql_query_chain.html) and [SQLDatabase](https://python.langchain.com/api_reference/community/utilities/langchain_community.utilities.sql_database.SQLDatabase.html), this is handled for you for any of the following dialects:

# In[2]:


from langchain.chains.sql_database.prompt import SQL_PROMPTS

list(SQL_PROMPTS)


# For example, using our current DB we can see that we'll get a SQLite-specific prompt.
# 
# import ChatModelTabs from "@theme/ChatModelTabs";
# 
# <ChatModelTabs customVarName="llm" />
# 

# In[3]:


# | output: false
# | echo: false

from langchain_openai import ChatOpenAI

llm = ChatOpenAI()


# In[4]:


from langchain.chains import create_sql_query_chain

chain = create_sql_query_chain(llm, db)
chain.get_prompts()[0].pretty_print()


# ## Table definitions and example rows
# 
# In most SQL chains, we'll need to feed the model at least part of the database schema. Without this it won't be able to write valid queries. Our database comes with some convenience methods to give us the relevant context. Specifically, we can get the table names, their schemas, and a sample of rows from each table.
# 
# Here we will use `SQLDatabase.get_context`, which provides available tables and their schemas:

# In[5]:


context = db.get_context()
print(list(context))
print(context["table_info"])


# When we don't have too many, or too wide of, tables, we can just insert the entirety of this information in our prompt:

# In[6]:


prompt_with_context = chain.get_prompts()[0].partial(table_info=context["table_info"])
print(prompt_with_context.pretty_repr()[:1500])


# When we do have database schemas that are too large to fit into our model's context window, we'll need to come up with ways of inserting only the relevant table definitions into the prompt based on the user input. For more on this head to the [Many tables, wide tables, high-cardinality feature](/docs/how_to/sql_large_db) guide.

# ## Few-shot examples
# 
# Including examples of natural language questions being converted to valid SQL queries against our database in the prompt will often improve model performance, especially for complex queries.
# 
# Let's say we have the following examples:

# In[7]:


examples = [
    {"input": "List all artists.", "query": "SELECT * FROM Artist;"},
    {
        "input": "Find all albums for the artist 'AC/DC'.",
        "query": "SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');",
    },
    {
        "input": "List all tracks in the 'Rock' genre.",
        "query": "SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');",
    },
    {
        "input": "Find the total duration of all tracks.",
        "query": "SELECT SUM(Milliseconds) FROM Track;",
    },
    {
        "input": "List all customers from Canada.",
        "query": "SELECT * FROM Customer WHERE Country = 'Canada';",
    },
    {
        "input": "How many tracks are there in the album with ID 5?",
        "query": "SELECT COUNT(*) FROM Track WHERE AlbumId = 5;",
    },
    {
        "input": "Find the total number of invoices.",
        "query": "SELECT COUNT(*) FROM Invoice;",
    },
    {
        "input": "List all tracks that are longer than 5 minutes.",
        "query": "SELECT * FROM Track WHERE Milliseconds > 300000;",
    },
    {
        "input": "Who are the top 5 customers by total purchase?",
        "query": "SELECT CustomerId, SUM(Total) AS TotalPurchase FROM Invoice GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;",
    },
    {
        "input": "Which albums are from the year 2000?",
        "query": "SELECT * FROM Album WHERE strftime('%Y', ReleaseDate) = '2000';",
    },
    {
        "input": "How many employees are there",
        "query": 'SELECT COUNT(*) FROM "Employee"',
    },
]


# We can create a few-shot prompt with them like so:

# In[8]:


from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")
prompt = FewShotPromptTemplate(
    examples=examples[:5],
    example_prompt=example_prompt,
    prefix="You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. Unless otherwise specificed, do not return more than {top_k} rows.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries.",
    suffix="User input: {input}\nSQL query: ",
    input_variables=["input", "top_k", "table_info"],
)


# In[9]:


print(prompt.format(input="How many artists are there?", top_k=3, table_info="foo"))


# ## Dynamic few-shot examples
# 
# If we have enough examples, we may want to only include the most relevant ones in the prompt, either because they don't fit in the model's context window or because the long tail of examples distracts the model. And specifically, given any input we want to include the examples most relevant to that input.
# 
# We can do just this using an ExampleSelector. In this case we'll use a [SemanticSimilarityExampleSelector](https://python.langchain.com/api_reference/core/example_selectors/langchain_core.example_selectors.semantic_similarity.SemanticSimilarityExampleSelector.html), which will store the examples in the vector database of our choosing. At runtime it will perform a similarity search between the input and our examples, and return the most semantically similar ones.
# 
# We default to OpenAI embeddings here, but you can swap them out for the model provider of your choice.

# In[10]:


from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=5,
    input_keys=["input"],
)


# In[11]:


example_selector.select_examples({"input": "how many artists are there?"})


# To use it, we can pass the ExampleSelector directly in to our FewShotPromptTemplate:

# In[12]:


prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. Unless otherwise specificed, do not return more than {top_k} rows.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries.",
    suffix="User input: {input}\nSQL query: ",
    input_variables=["input", "top_k", "table_info"],
)


# In[13]:


print(prompt.format(input="how many artists are there?", top_k=3, table_info="foo"))


# Trying it out, we see that the model identifies the relevant table:

# In[14]:


chain = create_sql_query_chain(llm, db, prompt)
chain.invoke({"question": "how many artists are there?"})

