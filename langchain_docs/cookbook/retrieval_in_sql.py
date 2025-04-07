#!/usr/bin/env python
# coding: utf-8

# # Incoporating semantic similarity in tabular databases
# 
# In this notebook we will cover how to run semantic search over a specific table column within a single SQL query, combining tabular query with RAG.
# 
# 
# ### Overall workflow
# 
# 1. Generating embeddings for a specific column
# 2. Storing the embeddings in a new column (if column has low cardinality, it's better to use another table containing unique values and their embeddings)
# 3. Querying using standard SQL queries with [PGVector](https://github.com/pgvector/pgvector) extension which allows using L2 distance (`<->`), Cosine distance (`<=>` or cosine similarity using `1 - <=>`) and Inner product (`<#>`)
# 4. Running standard SQL query
# 
# ### Requirements
# 
# We will need a PostgreSQL database with [pgvector](https://github.com/pgvector/pgvector) extension enabled. For this example, we will use a `Chinook` database using a local PostgreSQL server.

# In[1]:


import getpass
import os

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or getpass.getpass(
    "OpenAI API Key:"
)


# In[ ]:


from langchain.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI

CONNECTION_STRING = "postgresql+psycopg2://postgres:test@localhost:5432/vectordb"  # Replace with your own
db = SQLDatabase.from_uri(CONNECTION_STRING)


# ### Embedding the song titles

# For this example, we will run queries based on semantic meaning of song titles. In order to do this, let's start by adding a new column in the table for storing the embeddings:

# In[3]:


# db.run('ALTER TABLE "Track" ADD COLUMN "embeddings" vector;')


# Let's generate the embedding for each *track title* and store it as a new column in our "Track" table

# In[4]:


from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()


# In[5]:


tracks = db.run('SELECT "Name" FROM "Track"')
song_titles = [s[0] for s in eval(tracks)]
title_embeddings = embeddings_model.embed_documents(song_titles)
len(title_embeddings)


# Now let's insert the embeddings in the into the new column from our table

# In[6]:


from tqdm import tqdm

for i in tqdm(range(len(title_embeddings))):
    title = song_titles[i].replace("'", "''")
    embedding = title_embeddings[i]
    sql_command = (
        f'UPDATE "Track" SET "embeddings" = ARRAY{embedding} WHERE "Name" ='
        + f"'{title}'"
    )
    db.run(sql_command)


# We can test the semantic search running the following query:

# In[7]:


embeded_title = embeddings_model.embed_query("hope about the future")
query = (
    'SELECT "Track"."Name" FROM "Track" WHERE "Track"."embeddings" IS NOT NULL ORDER BY "embeddings" <-> '
    + f"'{embeded_title}' LIMIT 5"
)
db.run(query)


# ### Creating the SQL Chain

# Let's start by defining useful functions to get info from database and running the query:

# In[8]:


def get_schema(_):
    return db.get_table_info()


def run_query(query):
    return db.run(query)


# Now let's build the **prompt** we will use. This prompt is an extension from [text-to-postgres-sql](https://smith.langchain.com/hub/jacob/text-to-postgres-sql?organizationId=f9b614b8-5c3a-4e7c-afbc-6d7ad4fd8892) prompt

# In[9]:


from langchain_core.prompts import ChatPromptTemplate

template = """You are a Postgres expert. Given an input question, first create a syntactically correct Postgres query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause as per Postgres. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".

You can use an extra extension which allows you to run semantic similarity using <-> operator on tables containing columns named "embeddings".
<-> operator can ONLY be used on embeddings columns.
The embeddings value for a given row typically represents the semantic meaning of that row.
The vector represents an embedding representation of the question, given below. 
Do NOT fill in the vector values directly, but rather specify a `[search_word]` placeholder, which should contain the word that would be embedded for filtering.
For example, if the user asks for songs about 'the feeling of loneliness' the query could be:
'SELECT "[whatever_table_name]"."SongName" FROM "[whatever_table_name]" ORDER BY "embeddings" <-> '[loneliness]' LIMIT 5'

Use the following format:

Question: <Question here>
SQLQuery: <SQL Query to run>
SQLResult: <Result of the SQLQuery>
Answer: <Final answer here>

Only use the following tables:

{schema}
"""


prompt = ChatPromptTemplate.from_messages(
    [("system", template), ("human", "{question}")]
)


# And we can create the chain using **[LangChain Expression Language](https://python.langchain.com/docs/expression_language/)**:

# In[10]:


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

db = SQLDatabase.from_uri(
    CONNECTION_STRING
)  # We reconnect to db so the new columns are loaded as well.
llm = ChatOpenAI(model="gpt-4", temperature=0)

sql_query_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)


# In[11]:


sql_query_chain.invoke(
    {
        "question": "Which are the 5 rock songs with titles about deep feeling of dispair?"
    }
)


# This chain simply generates the query. Now we will create the full chain that also handles the execution and the final result for the user:

# In[12]:


import re

from langchain_core.runnables import RunnableLambda


def replace_brackets(match):
    words_inside_brackets = match.group(1).split(", ")
    embedded_words = [
        str(embeddings_model.embed_query(word)) for word in words_inside_brackets
    ]
    return "', '".join(embedded_words)


def get_query(query):
    sql_query = re.sub(r"\[([\w\s,]+)\]", replace_brackets, query)
    return sql_query


template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""

prompt = ChatPromptTemplate.from_messages(
    [("system", template), ("human", "{question}")]
)

full_chain = (
    RunnablePassthrough.assign(query=sql_query_chain)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=RunnableLambda(lambda x: db.run(get_query(x["query"]))),
    )
    | prompt
    | llm
)


# ## Using the Chain

# ### Example 1: Filtering a column based on semantic meaning

# Let's say we want to retrieve songs that express `deep feeling of dispair`, but filtering based on genre:

# In[11]:


full_chain.invoke(
    {
        "question": "Which are the 5 rock songs with titles about deep feeling of dispair?"
    }
)


# What is substantially different in implementing this method is that we have combined:
# - Semantic search (songs that have titles with some semantic meaning)
# - Traditional tabular querying (running JOIN statements to filter track based on genre)
# 
# This is something we _could_ potentially achieve using metadata filtering, but it's more complex to do so (we would need to use a vector database containing the embeddings, and use metadata filtering based on genre).
# 
# However, for other use cases metadata filtering **wouldn't be enough**.

# ### Example 2: Combining filters

# In[29]:


full_chain.invoke(
    {
        "question": "I want to know the 3 albums which have the most amount of songs in the top 150 saddest songs"
    }
)


# So we have result for 3 albums with most amount of songs in top 150 saddest ones. This **wouldn't** be possible using only standard metadata filtering. Without this _hybdrid query_, we would need some postprocessing to get the result.
# 
# Another similar exmaple:

# In[30]:


full_chain.invoke(
    {
        "question": "I need the 6 albums with shortest title, as long as they contain songs which are in the 20 saddest song list."
    }
)


# Let's see what the query looks like to double check:

# In[32]:


print(
    sql_query_chain.invoke(
        {
            "question": "I need the 6 albums with shortest title, as long as they contain songs which are in the 20 saddest song list."
        }
    )
)


# ### Example 3: Combining two separate semantic searches
# 
# One interesting aspect of this approach which is **substantially different from using standar RAG** is that we can even **combine** two semantic search filters:
# - _Get 5 saddest songs..._
# - _**...obtained from albums with "lovely" titles**_
# 
# This could generalize to **any kind of combined RAG** (paragraphs discussing _X_ topic belonging from books about _Y_, replies to a tweet about _ABC_ topic that express _XYZ_ feeling)
# 
# We will combine semantic search on songs and album titles, so we need to do the same for `Album` table:
# 1. Generate the embeddings
# 2. Add them to the table as a new column (which we need to add in the table)

# In[60]:


# db.run('ALTER TABLE "Album" ADD COLUMN "embeddings" vector;')


# In[43]:


albums = db.run('SELECT "Title" FROM "Album"')
album_titles = [title[0] for title in eval(albums)]
album_title_embeddings = embeddings_model.embed_documents(album_titles)
for i in tqdm(range(len(album_title_embeddings))):
    album_title = album_titles[i].replace("'", "''")
    album_embedding = album_title_embeddings[i]
    sql_command = (
        f'UPDATE "Album" SET "embeddings" = ARRAY{album_embedding} WHERE "Title" ='
        + f"'{album_title}'"
    )
    db.run(sql_command)


# In[45]:


embeded_title = embeddings_model.embed_query("hope about the future")
query = (
    'SELECT "Album"."Title" FROM "Album" WHERE "Album"."embeddings" IS NOT NULL ORDER BY "embeddings" <-> '
    + f"'{embeded_title}' LIMIT 5"
)
db.run(query)


# Now we can combine both filters:

# In[54]:


db = SQLDatabase.from_uri(
    CONNECTION_STRING
)  # We reconnect to dbso the new columns are loaded as well.


# In[49]:


full_chain.invoke(
    {
        "question": "I want to know songs about breakouts obtained from top 5 albums about love"
    }
)


# This is something **different** that **couldn't be achieved** using standard metadata filtering over a vectordb.
