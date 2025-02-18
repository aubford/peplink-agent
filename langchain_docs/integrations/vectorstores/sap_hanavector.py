#!/usr/bin/env python
# coding: utf-8

# # SAP HANA Cloud Vector Engine
#
# >[SAP HANA Cloud Vector Engine](https://www.sap.com/events/teched/news-guide/ai.html#article8) is a vector store fully integrated into the `SAP HANA Cloud` database.
#
# You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

# ## Setting up
#
# Installation of the HANA database driver.

# In[1]:


# Pip install necessary package
get_ipython().run_line_magic("pip", "install --upgrade --quiet  hdbcli")


# For `OpenAIEmbeddings` we use the OpenAI API key from the environment.

# In[1]:


import os

# Use OPENAI_API_KEY env variable
# os.environ["OPENAI_API_KEY"] = "Your OpenAI API key"


# Create a database connection to a HANA Cloud instance.

# In[9]:


from dotenv import load_dotenv
from hdbcli import dbapi

load_dotenv()
# Use connection settings from the environment
connection = dbapi.connect(
    address=os.environ.get("HANA_DB_ADDRESS"),
    port=os.environ.get("HANA_DB_PORT"),
    user=os.environ.get("HANA_DB_USER"),
    password=os.environ.get("HANA_DB_PASSWORD"),
    autocommit=True,
    sslValidateCertificate=False,
)


# ## Example

# Load the sample document "state_of_the_union.txt" and create chunks from it.

# In[10]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.hanavector import HanaDB
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

text_documents = TextLoader("../../how_to/state_of_the_union.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
text_chunks = text_splitter.split_documents(text_documents)
print(f"Number of document chunks: {len(text_chunks)}")

embeddings = OpenAIEmbeddings()


# Create a LangChain VectorStore interface for the HANA database and specify the table (collection) to use for accessing the vector embeddings

# In[11]:


db = HanaDB(
    embedding=embeddings, connection=connection, table_name="STATE_OF_THE_UNION"
)


# Add the loaded document chunks to the table. For this example, we delete any previous content from the table which might exist from previous runs.

# In[12]:


# Delete already existing documents from the table
db.delete(filter={})

# add the loaded document chunks
db.add_documents(text_chunks)


# Perform a query to get the two best-matching document chunks from the ones that were added in the previous step.
# By default "Cosine Similarity" is used for the search.

# In[13]:


query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query, k=2)

for doc in docs:
    print("-" * 80)
    print(doc.page_content)


# Query the same content with "Euclidian Distance". The results shoud be the same as with "Cosine Similarity".

# In[14]:


from langchain_community.vectorstores.utils import DistanceStrategy

db = HanaDB(
    embedding=embeddings,
    connection=connection,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    table_name="STATE_OF_THE_UNION",
)

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query, k=2)
for doc in docs:
    print("-" * 80)
    print(doc.page_content)


# ## Maximal Marginal Relevance Search (MMR)
#
# `Maximal marginal relevance` optimizes for similarity to query AND diversity among selected documents. The first 20 (fetch_k) items will be retrieved from the DB. The MMR algorithm will then find the best 2 (k) matches.

# In[15]:


docs = db.max_marginal_relevance_search(query, k=2, fetch_k=20)
for doc in docs:
    print("-" * 80)
    print(doc.page_content)


# ## Creating an HNSW Vector Index
#
# A vector index can significantly speed up top-k nearest neighbor queries for vectors. Users can create a Hierarchical Navigable Small World (HNSW) vector index using the `create_hnsw_index` function.
#
# For more information about creating an index at the database level, please refer to the [official documentation](https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-vector-engine-guide/create-vector-index-statement-data-definition).
#
#

# In[18]:


# HanaDB instance uses cosine similarity as default:
db_cosine = HanaDB(
    embedding=embeddings, connection=connection, table_name="STATE_OF_THE_UNION"
)

# Attempting to create the HNSW index with default parameters
db_cosine.create_hnsw_index()  # If no other parameters are specified, the default values will be used
# Default values: m=64, ef_construction=128, ef_search=200
# The default index name will be: STATE_OF_THE_UNION_COSINE_SIMILARITY_IDX (verify this naming pattern in HanaDB class)


# Creating a HanaDB instance with L2 distance as the similarity function and defined values
db_l2 = HanaDB(
    embedding=embeddings,
    connection=connection,
    table_name="STATE_OF_THE_UNION",
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,  # Specify L2 distance
)

# This will create an index based on L2 distance strategy.
db_l2.create_hnsw_index(
    index_name="STATE_OF_THE_UNION_L2_index",
    m=100,  # Max number of neighbors per graph node (valid range: 4 to 1000)
    ef_construction=200,  # Max number of candidates during graph construction (valid range: 1 to 100000)
    ef_search=500,  # Min number of candidates during the search (valid range: 1 to 100000)
)

# Use L2 index to perform MMR
docs = db_l2.max_marginal_relevance_search(query, k=2, fetch_k=20)
for doc in docs:
    print("-" * 80)
    print(doc.page_content)


#
# **Key Points**:
# - **Similarity Function**: The similarity function for the index is **cosine similarity** by default. If you want to use a different similarity function (e.g., `L2` distance), you need to specify it when initializing the `HanaDB` instance.
# - **Default Parameters**: In the `create_hnsw_index` function, if the user does not provide custom values for parameters like `m`, `ef_construction`, or `ef_search`, the default values (e.g., `m=64`, `ef_construction=128`, `ef_search=200`) will be used automatically. These values ensure the index is created with reasonable performance without requiring user intervention.
#
#

# ## Basic Vectorstore Operations

# In[19]:


db = HanaDB(
    connection=connection, embedding=embeddings, table_name="LANGCHAIN_DEMO_BASIC"
)

# Delete already existing documents from the table
db.delete(filter={})


# We can add simple text documents to the existing table.

# In[20]:


docs = [Document(page_content="Some text"), Document(page_content="Other docs")]
db.add_documents(docs)


# Add documents with metadata.

# In[21]:


docs = [
    Document(
        page_content="foo",
        metadata={"start": 100, "end": 150, "doc_name": "foo.txt", "quality": "bad"},
    ),
    Document(
        page_content="bar",
        metadata={"start": 200, "end": 250, "doc_name": "bar.txt", "quality": "good"},
    ),
]
db.add_documents(docs)


# Query documents with specific metadata.

# In[22]:


docs = db.similarity_search("foobar", k=2, filter={"quality": "bad"})
# With filtering on "quality"=="bad", only one document should be returned
for doc in docs:
    print("-" * 80)
    print(doc.page_content)
    print(doc.metadata)


# Delete documents with specific metadata.

# In[23]:


db.delete(filter={"quality": "bad"})

# Now the similarity search with the same filter will return no results
docs = db.similarity_search("foobar", k=2, filter={"quality": "bad"})
print(len(docs))


# ## Advanced filtering
# In addition to the basic value-based filtering capabilities, it is possible to use more advanced filtering.
# The table below shows the available filter operators.
#
# | Operator | Semantic                 |
# |----------|-------------------------|
# | `$eq`    | Equality (==)           |
# | `$ne`    | Inequality (!=)         |
# | `$lt`    | Less than (&lt;)           |
# | `$lte`   | Less than or equal (&lt;=) |
# | `$gt`    | Greater than (>)        |
# | `$gte`   | Greater than or equal (>=) |
# | `$in`    | Contained in a set of given values  (in)    |
# | `$nin`   | Not contained in a set of given values  (not in)  |
# | `$between` | Between the range of two boundary values |
# | `$like`  | Text equality based on the "LIKE" semantics in SQL (using "%" as wildcard)  |
# | `$and`   | Logical "and", supporting 2 or more operands |
# | `$or`    | Logical "or", supporting 2 or more operands |

# In[24]:


# Prepare some test documents
docs = [
    Document(
        page_content="First",
        metadata={"name": "adam", "is_active": True, "id": 1, "height": 10.0},
    ),
    Document(
        page_content="Second",
        metadata={"name": "bob", "is_active": False, "id": 2, "height": 5.7},
    ),
    Document(
        page_content="Third",
        metadata={"name": "jane", "is_active": True, "id": 3, "height": 2.4},
    ),
]

db = HanaDB(
    connection=connection,
    embedding=embeddings,
    table_name="LANGCHAIN_DEMO_ADVANCED_FILTER",
)

# Delete already existing documents from the table
db.delete(filter={})
db.add_documents(docs)


# Helper function for printing filter results
def print_filter_result(result):
    if len(result) == 0:
        print("<empty result>")
    for doc in result:
        print(doc.metadata)


# Filtering with `$ne`, `$gt`, `$gte`, `$lt`, `$lte`

# In[25]:


advanced_filter = {"id": {"$ne": 1}}
print(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search("just testing", k=5, filter=advanced_filter))

advanced_filter = {"id": {"$gt": 1}}
print(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search("just testing", k=5, filter=advanced_filter))

advanced_filter = {"id": {"$gte": 1}}
print(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search("just testing", k=5, filter=advanced_filter))

advanced_filter = {"id": {"$lt": 1}}
print(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search("just testing", k=5, filter=advanced_filter))

advanced_filter = {"id": {"$lte": 1}}
print(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search("just testing", k=5, filter=advanced_filter))


# Filtering with `$between`, `$in`, `$nin`

# In[26]:


advanced_filter = {"id": {"$between": (1, 2)}}
print(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search("just testing", k=5, filter=advanced_filter))

advanced_filter = {"name": {"$in": ["adam", "bob"]}}
print(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search("just testing", k=5, filter=advanced_filter))

advanced_filter = {"name": {"$nin": ["adam", "bob"]}}
print(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search("just testing", k=5, filter=advanced_filter))


# Text filtering with `$like`

# In[27]:


advanced_filter = {"name": {"$like": "a%"}}
print(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search("just testing", k=5, filter=advanced_filter))

advanced_filter = {"name": {"$like": "%a%"}}
print(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search("just testing", k=5, filter=advanced_filter))


# Combined filtering with `$and`, `$or`

# In[28]:


advanced_filter = {"$or": [{"id": 1}, {"name": "bob"}]}
print(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search("just testing", k=5, filter=advanced_filter))

advanced_filter = {"$and": [{"id": 1}, {"id": 2}]}
print(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search("just testing", k=5, filter=advanced_filter))

advanced_filter = {"$or": [{"id": 1}, {"id": 2}, {"id": 3}]}
print(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search("just testing", k=5, filter=advanced_filter))


# ## Using a VectorStore as a retriever in chains for retrieval augmented generation (RAG)

# In[29]:


from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# Access the vector DB with a new table
db = HanaDB(
    connection=connection,
    embedding=embeddings,
    table_name="LANGCHAIN_DEMO_RETRIEVAL_CHAIN",
)

# Delete already existing entries from the table
db.delete(filter={})

# add the loaded document chunks from the "State Of The Union" file
db.add_documents(text_chunks)

# Create a retriever instance of the vector store
retriever = db.as_retriever()


# Define the prompt.

# In[30]:


from langchain_core.prompts import PromptTemplate

prompt_template = """
You are an expert in state of the union topics. You are provided multiple context items that are related to the prompt you have to answer.
Use the following pieces of context to answer the question at the end.

'''
{context}
'''

Question: {question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}


# Create the ConversationalRetrievalChain, which handles the chat history and the retrieval of similar document chunks to be added to the prompt.

# In[ ]:


from langchain.chains import ConversationalRetrievalChain

llm = ChatOpenAI(model="gpt-3.5-turbo")
memory = ConversationBufferMemory(
    memory_key="chat_history", output_key="answer", return_messages=True
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    db.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    memory=memory,
    verbose=False,
    combine_docs_chain_kwargs={"prompt": PROMPT},
)


# Ask the first question (and verify how many text chunks have been used).

# In[32]:


question = "What about Mexico and Guatemala?"

result = qa_chain.invoke({"question": question})
print("Answer from LLM:")
print("================")
print(result["answer"])

source_docs = result["source_documents"]
print("================")
print(f"Number of used source document chunks: {len(source_docs)}")


# Examine the used chunks of the chain in detail. Check if the best ranked chunk contains info about "Mexico and Guatemala" as mentioned in the question.

# In[ ]:


for doc in source_docs:
    print("-" * 80)
    print(doc.page_content)
    print(doc.metadata)


# Ask another question on the same conversational chain. The answer should relate to the previous answer given.

# In[34]:


question = "What about other countries?"

result = qa_chain.invoke({"question": question})
print("Answer from LLM:")
print("================")
print(result["answer"])


# ## Standard tables vs. "custom" tables with vector data

# As default behaviour, the table for the embeddings is created with 3 columns:
#
# - A column `VEC_TEXT`, which contains the text of the Document
# - A column `VEC_META`, which contains the metadata of the Document
# - A column `VEC_VECTOR`, which contains the embeddings-vector of the Document's text

# In[35]:


# Access the vector DB with a new table
db = HanaDB(
    connection=connection, embedding=embeddings, table_name="LANGCHAIN_DEMO_NEW_TABLE"
)

# Delete already existing entries from the table
db.delete(filter={})

# Add a simple document with some metadata
docs = [
    Document(
        page_content="A simple document",
        metadata={"start": 100, "end": 150, "doc_name": "simple.txt"},
    )
]
db.add_documents(docs)


# Show the columns in table "LANGCHAIN_DEMO_NEW_TABLE"

# In[36]:


cur = connection.cursor()
cur.execute(
    "SELECT COLUMN_NAME, DATA_TYPE_NAME FROM SYS.TABLE_COLUMNS WHERE SCHEMA_NAME = CURRENT_SCHEMA AND TABLE_NAME = 'LANGCHAIN_DEMO_NEW_TABLE'"
)
rows = cur.fetchall()
for row in rows:
    print(row)
cur.close()


# Show the value of the inserted document in the three columns

# In[ ]:


cur = connection.cursor()
cur.execute(
    "SELECT VEC_TEXT, VEC_META, TO_NVARCHAR(VEC_VECTOR) FROM LANGCHAIN_DEMO_NEW_TABLE LIMIT 1"
)
rows = cur.fetchall()
print(rows[0][0])  # The text
print(rows[0][1])  # The metadata
print(rows[0][2])  # The vector
cur.close()


# Custom tables must have at least three columns that match the semantics of a standard table
#
# - A column with type `NCLOB` or `NVARCHAR` for the text/context of the embeddings
# - A column with type `NCLOB` or `NVARCHAR` for the metadata
# - A column with type `REAL_VECTOR` for the embedding vector
#
# The table can contain additional columns. When new Documents are inserted into the table, these additional columns must allow NULL values.

# In[39]:


# Create a new table "MY_OWN_TABLE_ADD" with three "standard" columns and one additional column
my_own_table_name = "MY_OWN_TABLE_ADD"
cur = connection.cursor()
cur.execute(
    (
        f"CREATE TABLE {my_own_table_name} ("
        "SOME_OTHER_COLUMN NVARCHAR(42), "
        "MY_TEXT NVARCHAR(2048), "
        "MY_METADATA NVARCHAR(1024), "
        "MY_VECTOR REAL_VECTOR )"
    )
)

# Create a HanaDB instance with the own table
db = HanaDB(
    connection=connection,
    embedding=embeddings,
    table_name=my_own_table_name,
    content_column="MY_TEXT",
    metadata_column="MY_METADATA",
    vector_column="MY_VECTOR",
)

# Add a simple document with some metadata
docs = [
    Document(
        page_content="Some other text",
        metadata={"start": 400, "end": 450, "doc_name": "other.txt"},
    )
]
db.add_documents(docs)

# Check if data has been inserted into our own table
cur.execute(f"SELECT * FROM {my_own_table_name} LIMIT 1")
rows = cur.fetchall()
print(rows[0][0])  # Value of column "SOME_OTHER_DATA". Should be NULL/None
print(rows[0][1])  # The text
print(rows[0][2])  # The metadata
print(rows[0][3])  # The vector

cur.close()


# Add another document and perform a similarity search on the custom table.

# In[40]:


docs = [
    Document(
        page_content="Some more text",
        metadata={"start": 800, "end": 950, "doc_name": "more.txt"},
    )
]
db.add_documents(docs)

query = "What's up?"
docs = db.similarity_search(query, k=2)
for doc in docs:
    print("-" * 80)
    print(doc.page_content)


# ### Filter Performance Optimization with Custom Columns

# To allow flexible metadata values, all metadata is stored as JSON in the metadata column by default. If some of the used metadata keys and value types are known, they can be stored in additional columns instead by creating the target table with the key names as column names and passing them to the HanaDB constructor via the specific_metadata_columns list. Metadata keys that match those values are copied into the special column during insert. Filters use the special columns instead of the metadata JSON column for keys in the specific_metadata_columns list.

# In[41]:


# Create a new table "PERFORMANT_CUSTOMTEXT_FILTER" with three "standard" columns and one additional column
my_own_table_name = "PERFORMANT_CUSTOMTEXT_FILTER"
cur = connection.cursor()
cur.execute(
    (
        f"CREATE TABLE {my_own_table_name} ("
        "CUSTOMTEXT NVARCHAR(500), "
        "MY_TEXT NVARCHAR(2048), "
        "MY_METADATA NVARCHAR(1024), "
        "MY_VECTOR REAL_VECTOR )"
    )
)

# Create a HanaDB instance with the own table
db = HanaDB(
    connection=connection,
    embedding=embeddings,
    table_name=my_own_table_name,
    content_column="MY_TEXT",
    metadata_column="MY_METADATA",
    vector_column="MY_VECTOR",
    specific_metadata_columns=["CUSTOMTEXT"],
)

# Add a simple document with some metadata
docs = [
    Document(
        page_content="Some other text",
        metadata={
            "start": 400,
            "end": 450,
            "doc_name": "other.txt",
            "CUSTOMTEXT": "Filters on this value are very performant",
        },
    )
]
db.add_documents(docs)

# Check if data has been inserted into our own table
cur.execute(f"SELECT * FROM {my_own_table_name} LIMIT 1")
rows = cur.fetchall()
print(
    rows[0][0]
)  # Value of column "CUSTOMTEXT". Should be "Filters on this value are very performant"
print(rows[0][1])  # The text
print(
    rows[0][2]
)  # The metadata without the "CUSTOMTEXT" data, as this is extracted into a sperate column
print(rows[0][3])  # The vector

cur.close()


# The special columns are completely transparent to the rest of the langchain interface. Everything works as it did before, just more performant.

# In[42]:


docs = [
    Document(
        page_content="Some more text",
        metadata={
            "start": 800,
            "end": 950,
            "doc_name": "more.txt",
            "CUSTOMTEXT": "Another customtext value",
        },
    )
]
db.add_documents(docs)

advanced_filter = {"CUSTOMTEXT": {"$like": "%value%"}}
query = "What's up?"
docs = db.similarity_search(query, k=2, filter=advanced_filter)
for doc in docs:
    print("-" * 80)
    print(doc.page_content)
