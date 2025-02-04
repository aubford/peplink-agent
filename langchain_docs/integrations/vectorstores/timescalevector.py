#!/usr/bin/env python
# coding: utf-8

# # Timescale Vector (Postgres)
# 
# >[Timescale Vector](https://www.timescale.com/ai?utm_campaign=vectorlaunch&utm_source=langchain&utm_medium=referral) is `PostgreSQL++` vector database for AI applications.
# 
# This notebook shows how to use the Postgres vector database `Timescale Vector`. You'll learn how to use TimescaleVector for (1) semantic search, (2) time-based vector search, (3) self-querying, and (4) how to create indexes to speed up queries.
# 
# ## What is Timescale Vector?
# 
# `Timescale Vector` enables you to efficiently store and query millions of vector embeddings in `PostgreSQL`.
# - Enhances `pgvector` with faster and more accurate similarity search on 100M+ vectors via `DiskANN` inspired indexing algorithm.
# - Enables fast time-based vector search via automatic time-based partitioning and indexing.
# - Provides a familiar SQL interface for querying vector embeddings and relational data.
# 
# `Timescale Vector` is cloud `PostgreSQL` for AI that scales with you from POC to production:
# - Simplifies operations by enabling you to store relational metadata, vector embeddings, and time-series data in a single database.
# - Benefits from rock-solid PostgreSQL foundation with enterprise-grade features like streaming backups and replication, high availability and row-level security.
# - Enables a worry-free experience with enterprise-grade security and compliance.
# 
# ## How to access Timescale Vector
# 
# `Timescale Vector` is available on [Timescale](https://www.timescale.com/ai?utm_campaign=vectorlaunch&utm_source=langchain&utm_medium=referral), the cloud PostgreSQL platform. (There is no self-hosted version at this time.)
# 
# LangChain users get a 90-day free trial for Timescale Vector.
# - To get started, [signup](https://console.cloud.timescale.com/signup?utm_campaign=vectorlaunch&utm_source=langchain&utm_medium=referral) to Timescale, create a new database and follow this notebook!
# - See the [Timescale Vector explainer blog](https://www.timescale.com/blog/how-we-made-postgresql-the-best-vector-database/?utm_campaign=vectorlaunch&utm_source=langchain&utm_medium=referral) for more details and performance benchmarks.
# - See the [installation instructions](https://github.com/timescale/python-vector) for more details on using Timescale Vector in Python.

# ## Setup
# 
# Follow these steps to get ready to follow this tutorial.

# In[ ]:


# Pip install necessary packages
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  timescale-vector')
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  langchain-openai langchain-community')
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  tiktoken')


# In this example, we'll use `OpenAIEmbeddings`, so let's load your OpenAI API key.

# In[1]:


import os

# Run export OPENAI_API_KEY=sk-YOUR_OPENAI_API_KEY...
# Get openAI api key by reading local .env file
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


# In[ ]:


# Get the API key and save it as an environment variable
# import os
# import getpass
# os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")


# In[2]:


from typing import Tuple


# Next we'll import the needed Python libraries and libraries from LangChain. Note that we import the `timescale-vector` library as well as the TimescaleVector LangChain vectorstore.

# In[3]:


from datetime import datetime, timedelta

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.vectorstores.timescalevector import TimescaleVector
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


# ## 1. Similarity Search with Euclidean Distance (Default)
# 
# First, we'll look at an example of doing a similarity search query on the State of the Union speech to find the most similar sentences to a given query sentence. We'll use the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) as our similarity metric.

# In[4]:


# Load the text and split it into chunks
loader = TextLoader("../../../extras/modules/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


# Next, we'll load the service URL for our Timescale database. 
# 
# If you haven't already, [signup for Timescale](https://console.cloud.timescale.com/signup?utm_campaign=vectorlaunch&utm_source=langchain&utm_medium=referral), and create a new database.
# 
# Then, to connect to your PostgreSQL database, you'll need your service URI, which can be found in the cheatsheet or `.env` file you downloaded after creating a new database. 
# 
# The URI will look something like this: `postgres://tsdbadmin:<password>@<id>.tsdb.cloud.timescale.com:<port>/tsdb?sslmode=require`. 

# In[5]:


# Timescale Vector needs the service url to your cloud database. You can see this as soon as you create the
# service in the cloud UI or in your credentials.sql file
SERVICE_URL = os.environ["TIMESCALE_SERVICE_URL"]

# Specify directly if testing
# SERVICE_URL = "postgres://tsdbadmin:<password>@<id>.tsdb.cloud.timescale.com:<port>/tsdb?sslmode=require"

# # You can get also it from an environment variables. We suggest using a .env file.
# import os
# SERVICE_URL = os.environ.get("TIMESCALE_SERVICE_URL", "")


# Next we create a TimescaleVector vectorstore. We specify a collection name, which will be the name of the table our data is stored in. 
# 
# Note: When creating a new instance of TimescaleVector, the TimescaleVector Module will try to create a table with the name of the collection. So, make sure that the collection name is unique (i.e it doesn't already exist).

# In[6]:


# The TimescaleVector Module will create a table with the name of the collection.
COLLECTION_NAME = "state_of_the_union_test"

# Create a Timescale Vector instance from the collection of documents
db = TimescaleVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    service_url=SERVICE_URL,
)


# Now that we've loaded our data, we can perform a similarity search.

# In[7]:


query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = db.similarity_search_with_score(query)


# In[8]:


for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)


# ### Using a Timescale Vector as a Retriever
# After initializing a TimescaleVector store, you can use it as a [retriever](/docs/how_to#retrievers).

# In[9]:


# Use TimescaleVector as a retriever
retriever = db.as_retriever()


# In[10]:


print(retriever)


# Let's look at an example of using Timescale Vector as a retriever with the RetrievalQA chain and the stuff documents chain.
# 
# In this example, we'll ask the same query as above, but this time we'll pass the relevant documents returned from Timescale Vector to an LLM to use as context to answer our question.
# 
# First we'll create our stuff chain:

# In[11]:


# Initialize GPT3.5 model
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo-16k")

# Initialize a RetrievalQA class from a stuff chain
from langchain.chains import RetrievalQA

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
)


# In[12]:


query = "What did the president say about Ketanji Brown Jackson?"
response = qa_stuff.run(query)


# In[13]:


print(response)


# ## 2. Similarity Search with time-based filtering
# 
# A key use case for Timescale Vector is efficient time-based vector search. Timescale Vector enables this by automatically partitioning vectors (and associated metadata) by time. This allows you to efficiently query vectors by both similarity to a query vector and time.
# 
# Time-based vector search functionality is helpful for applications like:
# - Storing and retrieving LLM response history (e.g. chatbots)
# - Finding the most recent embeddings that are similar to a query vector (e.g recent news).
# - Constraining similarity search to a relevant time range (e.g asking time-based questions about a knowledge base)
# 
# To illustrate how to use TimescaleVector's time-based vector search functionality, we'll ask questions about the git log history for TimescaleDB . We'll illustrate how to add documents with a time-based uuid and how run similarity searches with time range filters.

# ### Extract content and metadata from git log JSON
# First lets load in the git log data into a new collection in our PostgreSQL database named `timescale_commits`.

# In[14]:





# We'll define a helper funciton to create a uuid for a document and associated vector embedding based on its timestamp. We'll use this function to create a uuid for each git log entry.
# 
# Important note: If you are working with documents and want the current date and time associated with vector for time-based search, you can skip this step. A uuid will be automatically generated when the documents are ingested by default.

# In[15]:


from timescale_vector import client


# Function to take in a date string in the past and return a uuid v1
def create_uuid(date_string: str):
    if date_string is None:
        return None
    time_format = "%a %b %d %H:%M:%S %Y %z"
    datetime_obj = datetime.strptime(date_string, time_format)
    uuid = client.uuid_from_time(datetime_obj)
    return str(uuid)


# Next, we'll define a metadata function to extract the relevant metadata from the JSON record. We'll pass this function to the JSONLoader. See the [JSON document loader docs](/docs/how_to/document_loader_json) for more details.

# In[16]:


# Helper function to split name and email given an author string consisting of Name Lastname <email>
def split_name(input_string: str) -> Tuple[str, str]:
    if input_string is None:
        return None, None
    start = input_string.find("<")
    end = input_string.find(">")
    name = input_string[:start].strip()
    email = input_string[start + 1 : end].strip()
    return name, email


# Helper function to transform a date string into a timestamp_tz string
def create_date(input_string: str) -> datetime:
    if input_string is None:
        return None
    # Define a dictionary to map month abbreviations to their numerical equivalents
    month_dict = {
        "Jan": "01",
        "Feb": "02",
        "Mar": "03",
        "Apr": "04",
        "May": "05",
        "Jun": "06",
        "Jul": "07",
        "Aug": "08",
        "Sep": "09",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12",
    }

    # Split the input string into its components
    components = input_string.split()
    # Extract relevant information
    day = components[2]
    month = month_dict[components[1]]
    year = components[4]
    time = components[3]
    timezone_offset_minutes = int(components[5])  # Convert the offset to minutes
    timezone_hours = timezone_offset_minutes // 60  # Calculate the hours
    timezone_minutes = timezone_offset_minutes % 60  # Calculate the remaining minutes
    # Create a formatted string for the timestamptz in PostgreSQL format
    timestamp_tz_str = (
        f"{year}-{month}-{day} {time}+{timezone_hours:02}{timezone_minutes:02}"
    )
    return timestamp_tz_str


# Metadata extraction function to extract metadata from a JSON record
def extract_metadata(record: dict, metadata: dict) -> dict:
    record_name, record_email = split_name(record["author"])
    metadata["id"] = create_uuid(record["date"])
    metadata["date"] = create_date(record["date"])
    metadata["author_name"] = record_name
    metadata["author_email"] = record_email
    metadata["commit_hash"] = record["commit"]
    return metadata


# Next, you'll need to [download the sample dataset](https://s3.amazonaws.com/assets.timescale.com/ai/ts_git_log.json) and place it in the same directory as this notebook.
# 
# You can use following command:

# In[ ]:


# Download the file using curl and save it as commit_history.csv
# Note: Execute this command in your terminal, in the same directory as the notebook
get_ipython().system('curl -O https://s3.amazonaws.com/assets.timescale.com/ai/ts_git_log.json')


# Finally we can initialize the JSON loader to parse the JSON records. We also remove empty records for simplicity.

# In[17]:


# Define path to the JSON file relative to this notebook
# Change this to the path to your JSON file
FILE_PATH = "../../../../../ts_git_log.json"

# Load data from JSON file and extract metadata
loader = JSONLoader(
    file_path=FILE_PATH,
    jq_schema=".commit_history[]",
    text_content=False,
    metadata_func=extract_metadata,
)
documents = loader.load()

# Remove documents with None dates
documents = [doc for doc in documents if doc.metadata["date"] is not None]


# In[18]:


print(documents[0])


# ### Load documents and metadata into TimescaleVector vectorstore
# Now that we have prepared our documents, let's process them and load them, along with their vector embedding representations into our TimescaleVector vectorstore.
# 
# Since this is a demo, we will only load the first 500 records. In practice, you can load as many records as you want.

# In[19]:


NUM_RECORDS = 500
documents = documents[:NUM_RECORDS]


# Then we use the CharacterTextSplitter to split the documents into smaller chunks if needed for easier embedding. Note that this splitting process retains the metadata for each document.

# In[20]:


# Split the documents into chunks for embedding
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
docs = text_splitter.split_documents(documents)


# Next we'll create a Timescale Vector instance from the collection of documents that we finished pre-processsing.
# 
# First, we'll define a collection name, which will be the name of our table in the PostgreSQL database. 
# 
# We'll also define a time delta, which we pass to the `time_partition_interval` argument, which will be used to as the interval for partitioning the data by time. Each partition will consist of data for the specified length of time. We'll use 7 days for simplicity, but you can pick whatever value make sense for your use case -- for example if you query recent vectors frequently you might want to use a smaller time delta like 1 day, or if you query vectors over a decade long time period then you might want to use a larger time delta like 6 months or 1 year.
# 
# Finally, we'll create the TimescaleVector instance. We specify the `ids` argument to be the `uuid` field in our metadata that we created in the pre-processing step above. We do this because we want the time part of our uuids to reflect dates in the past (i.e when the commit was made). However, if we wanted the current date and time to be associated with our document, we can remove the id argument and uuid's will be automatically created with the current date and time.

# In[21]:


# Define collection name
COLLECTION_NAME = "timescale_commits"
embeddings = OpenAIEmbeddings()

# Create a Timescale Vector instance from the collection of documents
db = TimescaleVector.from_documents(
    embedding=embeddings,
    ids=[doc.metadata["id"] for doc in docs],
    documents=docs,
    collection_name=COLLECTION_NAME,
    service_url=SERVICE_URL,
    time_partition_interval=timedelta(days=7),
)


# ### Querying vectors by time and similarity
# 
# Now that we have loaded our documents into TimescaleVector, we can query them by time and similarity.
# 
# TimescaleVector provides multiple methods for querying vectors by doing similarity search with time-based filtering.
# 
# Let's take a look at each method below:

# In[31]:


# Time filter variables
start_dt = datetime(2023, 8, 1, 22, 10, 35)  # Start date = 1 August 2023, 22:10:35
end_dt = datetime(2023, 8, 30, 22, 10, 35)  # End date = 30 August 2023, 22:10:35
td = timedelta(days=7)  # Time delta = 7 days

query = "What's new with TimescaleDB functions?"


# Method 1: Filter within a provided start date and end date.
# 

# In[32]:


# Method 1: Query for vectors between start_date and end_date
docs_with_score = db.similarity_search_with_score(
    query, start_date=start_dt, end_date=end_dt
)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print("Date: ", doc.metadata["date"])
    print(doc.page_content)
    print("-" * 80)


# Note how the query only returns results within the specified date range.

# Method 2: Filter within a provided start date, and a time delta later.

# In[33]:


# Method 2: Query for vectors between start_dt and a time delta td later
# Most relevant vectors between 1 August and 7 days later
docs_with_score = db.similarity_search_with_score(
    query, start_date=start_dt, time_delta=td
)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print("Date: ", doc.metadata["date"])
    print(doc.page_content)
    print("-" * 80)


# Once again, notice how we get results within the specified time filter, different from the previous query.

# Method 3: Filter within a provided end date and a time delta earlier.

# In[34]:


# Method 3: Query for vectors between end_dt and a time delta td earlier
# Most relevant vectors between 30 August and 7 days earlier
docs_with_score = db.similarity_search_with_score(query, end_date=end_dt, time_delta=td)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print("Date: ", doc.metadata["date"])
    print(doc.page_content)
    print("-" * 80)


# Method 4: We can also filter for all vectors after a given date by only specifying a start date in our query.
# 
# Method 5: Similarly, we can filter for or all vectors before a given date by only specify an end date in our query.

# In[35]:


# Method 4: Query all vectors after start_date
docs_with_score = db.similarity_search_with_score(query, start_date=start_dt)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print("Date: ", doc.metadata["date"])
    print(doc.page_content)
    print("-" * 80)


# In[36]:


# Method 5: Query all vectors before end_date
docs_with_score = db.similarity_search_with_score(query, end_date=end_dt)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print("Date: ", doc.metadata["date"])
    print(doc.page_content)
    print("-" * 80)


# The main takeaway is that in each result above, only vectors within the specified time range are returned. These queries are very efficient as they only need to search the relevant partitions.

# We can also use this functionality for question answering, where we want to find the most relevant vectors within a specified time range to use as context for answering a question. Let's take a look at an example below, using Timescale Vector as a retriever:

# In[39]:


# Set timescale vector as a retriever and specify start and end dates via kwargs
retriever = db.as_retriever(search_kwargs={"start_date": start_dt, "end_date": end_dt})


# In[42]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo-16k")

from langchain.chains import RetrievalQA

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
)

query = (
    "What's new with the timescaledb functions? Tell me when these changes were made."
)
response = qa_stuff.run(query)
print(response)


# Note that the context the LLM uses to compose an answer are from retrieved documents only within the specified date range. 
# 
# This shows how you can use Timescale Vector to enhance retrieval augmented generation by retrieving documents within time ranges relevant to your query.

# ## 3. Using ANN Search Indexes to Speed Up Queries
# 
# You can speed up similarity queries by creating an index on the embedding column. You should only do this once you have ingested a large part of your data.
# 
# Timescale Vector supports the following indexes:
# - timescale_vector index (tsv): a disk-ann inspired graph index for fast similarity search (default).
# - pgvector's HNSW index: a hierarchical navigable small world graph index for fast similarity search.
# - pgvector's IVFFLAT index: an inverted file index for fast similarity search.
# 
# Important note: In PostgreSQL, each table can only have one index on a particular column. So if you'd like to test the performance of different index types, you can do so either by (1) creating multiple tables with different indexes, (2) creating multiple vector columns in the same table and creating different indexes on each column, or (3) by dropping and recreating the index on the same column and comparing results.

# In[43]:


# Initialize an existing TimescaleVector store
COLLECTION_NAME = "timescale_commits"
embeddings = OpenAIEmbeddings()
db = TimescaleVector(
    collection_name=COLLECTION_NAME,
    service_url=SERVICE_URL,
    embedding_function=embeddings,
)


# Using the `create_index()` function without additional arguments will create a timescale_vector_index by default, using the default parameters.

# In[44]:


# create an index
# by default this will create a Timescale Vector (DiskANN) index
db.create_index()


# You can also specify the parameters for the index. See the Timescale Vector documentation for a full discussion of the different parameters and their effects on performance.
# 
# Note: You don't need to specify parameters as we set smart defaults. But you can always specify your own parameters if you want to experiment eek out more performance for your specific dataset.

# In[45]:


# drop the old index
db.drop_index()

# create an index
# Note: You don't need to specify m and ef_construction parameters as we set smart defaults.
db.create_index(index_type="tsv", max_alpha=1.0, num_neighbors=50)


# 
# Timescale Vector also supports the HNSW ANN indexing algorithm, as well as the ivfflat ANN indexing algorithm. Simply specify in the `index_type` argument which index you'd like to create, and optionally specify the parameters for the index.

# In[46]:


# drop the old index
db.drop_index()

# Create an HNSW index
# Note: You don't need to specify m and ef_construction parameters as we set smart defaults.
db.create_index(index_type="hnsw", m=16, ef_construction=64)


# In[47]:


# drop the old index
db.drop_index()

# Create an IVFFLAT index
# Note: You don't need to specify num_lists and num_records parameters as we set smart defaults.
db.create_index(index_type="ivfflat", num_lists=20, num_records=1000)


# In general, we recommend using the default timescale vector index, or the HNSW index.

# In[48]:


# drop the old index
db.drop_index()
# Create a new timescale vector index
db.create_index()


# ## 4. Self Querying Retriever with Timescale Vector
# 
# Timescale Vector also supports the self-querying retriever functionality, which gives it the ability to query itself. Given a natural language query with a query statement and filters (single or composite), the retriever uses a query constructing LLM chain to write a SQL query and then applies it to the underlying PostgreSQL database in the Timescale Vector vectorstore.
# 
# For more on self-querying, [see the docs](/docs/how_to/self_query).

# To illustrate self-querying with Timescale Vector, we'll use the same gitlog dataset from Part 3.

# In[49]:


COLLECTION_NAME = "timescale_commits"
vectorstore = TimescaleVector(
    embedding_function=OpenAIEmbeddings(),
    collection_name=COLLECTION_NAME,
    service_url=SERVICE_URL,
)


# Next we'll create our self-querying retriever. To do this we'll need to provide some information upfront about the metadata fields that our documents support and a short description of the document contents.

# In[50]:


from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import OpenAI

# Give LLM info about the metadata fields
metadata_field_info = [
    AttributeInfo(
        name="id",
        description="A UUID v1 generated from the date of the commit",
        type="uuid",
    ),
    AttributeInfo(
        name="date",
        description="The date of the commit in timestamptz format",
        type="timestamptz",
    ),
    AttributeInfo(
        name="author_name",
        description="The name of the author of the commit",
        type="string",
    ),
    AttributeInfo(
        name="author_email",
        description="The email address of the author of the commit",
        type="string",
    ),
]
document_content_description = "The git log commit summary containing the commit hash, author, date of commit, change summary and change details"

# Instantiate the self-query retriever from an LLM
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    verbose=True,
)


# Now let's test out the self-querying retriever on our gitlog dataset. 
# 
# Run the queries below and note how you can specify a query, query with a filter, and query with a composite filter (filters with AND, OR) in natural language and the self-query retriever will translate that query into SQL and perform the search on the Timescale Vector PostgreSQL vectorstore.
# 
# This illustrates the power of the self-query retriever. You can use it to perform complex searches over your vectorstore without you or your users having to write any SQL directly!

# In[51]:


# This example specifies a relevant query
retriever.invoke("What are improvements made to continuous aggregates?")


# In[52]:


# This example specifies a filter
retriever.invoke("What commits did Sven Klemm add?")


# In[53]:


# This example specifies a query and filter
retriever.invoke("What commits about timescaledb_functions did Sven Klemm add?")


# In[54]:


# This example specifies a time-based filter
retriever.invoke("What commits were added in July 2023?")


# In[55]:


# This example specifies a query and a LIMIT value
retriever.invoke("What are two commits about hierarchical continuous aggregates?")


# ## 5. Working with an existing TimescaleVector vectorstore
# 
# In the examples above, we created a vectorstore from a collection of documents. However, often we want to work insert data into and query data from an existing vectorstore. Let's see how to initialize, add documents to, and query an existing collection of documents in a TimescaleVector vector store.
# 
# To work with an existing Timescale Vector store, we need to know the name of the table we want to query (`COLLECTION_NAME`) and the URL of the cloud PostgreSQL database (`SERVICE_URL`).

# In[56]:


# Initialize the existing
COLLECTION_NAME = "timescale_commits"
embeddings = OpenAIEmbeddings()
vectorstore = TimescaleVector(
    collection_name=COLLECTION_NAME,
    service_url=SERVICE_URL,
    embedding_function=embeddings,
)


# To load new data into the table, we use the `add_document()` function. This function takes a list of documents and a list of metadata. The metadata must contain a unique id for each document. 
# 
# If you want your documents to be associated with the current date and time, you do not need to create a list of ids. A uuid will be automatically generated for each document.
# 
# If you want your documents to be associated with a past date and time, you can create a list of ids using the `uuid_from_time` function in the `timecale-vector` python library, as shown in Section 2 above. This function takes a datetime object and returns a uuid with the date and time encoded in the uuid.

# In[58]:


# Add documents to a collection in TimescaleVector
ids = vectorstore.add_documents([Document(page_content="foo")])
ids


# In[59]:


# Query the vectorstore for similar documents
docs_with_score = vectorstore.similarity_search_with_score("foo")


# In[60]:


docs_with_score[0]


# In[61]:


docs_with_score[1]


# ### Deleting Data 
# 
# You can delete data by uuid or by a filter on the metadata.

# In[64]:


ids = vectorstore.add_documents([Document(page_content="Bar")])

vectorstore.delete(ids)


# Deleting using metadata is especially useful if you want to periodically update information scraped from a particular source, or particular date or some other metadata attribute.

# In[65]:


vectorstore.add_documents(
    [Document(page_content="Hello World", metadata={"source": "www.example.com/hello"})]
)
vectorstore.add_documents(
    [Document(page_content="Adios", metadata={"source": "www.example.com/adios"})]
)

vectorstore.delete_by_metadata({"source": "www.example.com/adios"})

vectorstore.add_documents(
    [
        Document(
            page_content="Adios, but newer!",
            metadata={"source": "www.example.com/adios"},
        )
    ]
)


# ### Overriding a vectorstore
# 
# If you have an existing collection, you override it by doing `from_documents` and setting `pre_delete_collection` = True

# In[ ]:


db = TimescaleVector.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    service_url=SERVICE_URL,
    pre_delete_collection=True,
)


# In[ ]:


docs_with_score = db.similarity_search_with_score("foo")


# In[ ]:


docs_with_score[0]

