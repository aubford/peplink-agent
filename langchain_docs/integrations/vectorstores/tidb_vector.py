#!/usr/bin/env python
# coding: utf-8

# # TiDB Vector
#
# > [TiDB Cloud](https://www.pingcap.com/tidb-serverless), is a comprehensive Database-as-a-Service (DBaaS) solution, that provides dedicated and serverless options. TiDB Serverless is now integrating a built-in vector search into the MySQL landscape. With this enhancement, you can seamlessly develop AI applications using TiDB Serverless without the need for a new database or additional technical stacks. Create a free TiDB Serverless cluster and start using the vector search feature at https://pingcap.com/ai.
#
# This notebook provides a detailed guide on utilizing the TiDB Vector functionality, showcasing its features and practical applications.

# ## Setting up environments

# Begin by installing the necessary packages.

# In[ ]:


get_ipython().run_line_magic("pip", "install langchain langchain-community")
get_ipython().run_line_magic("pip", "install langchain-openai")
get_ipython().run_line_magic("pip", "install pymysql")
get_ipython().run_line_magic("pip", "install tidb-vector")


# Configure both the OpenAI and TiDB host settings that you will need. In this notebook, we will follow the standard connection method provided by TiDB Cloud to establish a secure and efficient database connection.

# In[2]:


# Here we useimport getpass
import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
# copy from tidb cloud console
tidb_connection_string_template = "mysql+pymysql://<USER>:<PASSWORD>@<HOST>:4000/<DB>?ssl_ca=/etc/ssl/cert.pem&ssl_verify_cert=true&ssl_verify_identity=true"
# tidb_connection_string_template = "mysql+pymysql://root:<PASSWORD>@34.212.137.91:4000/test"
tidb_password = getpass.getpass("Input your TiDB password:")
tidb_connection_string = tidb_connection_string_template.replace(
    "<PASSWORD>", tidb_password
)


# Prepare the following data

# In[3]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import TiDBVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


# In[4]:


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


# ## Semantic similarity search
#
# TiDB supports both cosine and Euclidean distances ('cosine', 'l2'), with 'cosine' being the default choice.
#
# The code snippet below creates a table named `TABLE_NAME` in TiDB, optimized for vector searching. Upon successful execution of this code, you will be able to view and access the `TABLE_NAME` table directly within your TiDB database.

# In[5]:


TABLE_NAME = "semantic_embeddings"
db = TiDBVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    table_name=TABLE_NAME,
    connection_string=tidb_connection_string,
    distance_strategy="cosine",  # default, another option is "l2"
)


# In[6]:


query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = db.similarity_search_with_score(query, k=3)


# Please note that a lower cosine distance indicates higher similarity.

# In[7]:


for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)


# Additionally, the similarity_search_with_relevance_scores method can be used to obtain relevance scores, where a higher score indicates greater similarity.

# In[8]:


docs_with_relevance_score = db.similarity_search_with_relevance_scores(query, k=2)
for doc, score in docs_with_relevance_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)


# # Filter with metadata
#
# perform searches using metadata filters to retrieve a specific number of nearest-neighbor results that align with the applied filters.
#
# ## Supported metadata types
#
# Each vector in the TiDB Vector Store can be paired with metadata, structured as key-value pairs within a JSON object. The keys are strings, and the values can be of the following types:
#
# - String
# - Number (integer or floating point)
# - Booleans (true, false)
#
# For instance, consider the following valid metadata payloads:
#
# ```json
# {
#     "page": 12,
#     "book_tile": "Siddhartha"
# }
# ```
#
# ## Metadata filter syntax
#
# The available filters include:
#
# - $or - Selects vectors that meet any one of the given conditions.
# - $and - Selects vectors that meet all of the given conditions.
# - $eq - Equal to
# - $ne - Not equal to
# - $gt - Greater than
# - $gte - Greater than or equal to
# - $lt - Less than
# - $lte - Less than or equal to
# - $in - In array
# - $nin - Not in array
#
# Assuming one vector with metada:
# ```json
# {
#     "page": 12,
#     "book_tile": "Siddhartha"
# }
# ```
#
# The following metadata filters will match the vector
#
# ```json
# {"page": 12}
#
# {"page":{"$eq": 12}}
#
# {"page":{"$in": [11, 12, 13]}}
#
# {"page":{"$nin": [13]}}
#
# {"page":{"$lt": 11}}
#
# {
#     "$or": [{"page": 11}, {"page": 12}],
#     "$and": [{"page": 12}, {"page": 13}],
# }
# ```
#
# Please note that each key-value pair in the metadata filters is treated as a separate filter clause, and these clauses are combined using the AND logical operator.

# In[9]:


db.add_texts(
    texts=[
        "TiDB Vector offers advanced, high-speed vector processing capabilities, enhancing AI workflows with efficient data handling and analytics support.",
        "TiDB Vector, starting as low as $10 per month for basic usage",
    ],
    metadatas=[
        {"title": "TiDB Vector functionality"},
        {"title": "TiDB Vector Pricing"},
    ],
)


# In[10]:


docs_with_score = db.similarity_search_with_score(
    "Introduction to TiDB Vector", filter={"title": "TiDB Vector functionality"}, k=4
)
for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)


# ### Using as a Retriever

# In Langchain, a retriever is an interface that retrieves documents in response to an unstructured query, offering a broader functionality than a vector store. The code below demonstrates how to utilize TiDB Vector as a retriever.

# In[11]:


retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.8},
)
docs_retrieved = retriever.invoke(query)
for doc in docs_retrieved:
    print("-" * 80)
    print(doc.page_content)
    print("-" * 80)


# ## Advanced Use Case Scenario
#
# Let's look a advanced use case - a travel agent is crafting a custom travel report for clients who desire airports with specific amenities such as clean lounges and vegetarian options. The process involves:
# - A semantic search within airport reviews to extract airport codes meeting these amenities.
# - A subsequent SQL query that joins these codes with route information, detailing airlines and destinations aligned with the clients' preferences.
#
# First, let's prepare some airpod related data

# In[12]:


# create table to store airplan data
db.tidb_vector_client.execute(
    """CREATE TABLE airplan_routes (
        id INT AUTO_INCREMENT PRIMARY KEY,
        airport_code VARCHAR(10),
        airline_code VARCHAR(10),
        destination_code VARCHAR(10),
        route_details TEXT,
        duration TIME,
        frequency INT,
        airplane_type VARCHAR(50),
        price DECIMAL(10, 2),
        layover TEXT
    );"""
)

# insert some data into Routes and our vector table
db.tidb_vector_client.execute(
    """INSERT INTO airplan_routes (
        airport_code,
        airline_code,
        destination_code,
        route_details,
        duration,
        frequency,
        airplane_type,
        price,
        layover
    ) VALUES 
    ('JFK', 'DL', 'LAX', 'Non-stop from JFK to LAX.', '06:00:00', 5, 'Boeing 777', 299.99, 'None'),
    ('LAX', 'AA', 'ORD', 'Direct LAX to ORD route.', '04:00:00', 3, 'Airbus A320', 149.99, 'None'),
    ('EFGH', 'UA', 'SEA', 'Daily flights from SFO to SEA.', '02:30:00', 7, 'Boeing 737', 129.99, 'None');
    """
)
db.add_texts(
    texts=[
        "Clean lounges and excellent vegetarian dining options. Highly recommended.",
        "Comfortable seating in lounge areas and diverse food selections, including vegetarian.",
        "Small airport with basic facilities.",
    ],
    metadatas=[
        {"airport_code": "JFK"},
        {"airport_code": "LAX"},
        {"airport_code": "EFGH"},
    ],
)


# Finding Airports with Clean Facilities and Vegetarian Options via Vector Search

# In[13]:


retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.85},
)
semantic_query = "Could you recommend a US airport with clean lounges and good vegetarian dining options?"
reviews = retriever.invoke(semantic_query)
for r in reviews:
    print("-" * 80)
    print(r.page_content)
    print(r.metadata)
    print("-" * 80)


# In[14]:


# Extracting airport codes from the metadata
airport_codes = [review.metadata["airport_code"] for review in reviews]

# Executing a query to get the airport details
search_query = "SELECT * FROM airplan_routes WHERE airport_code IN :codes"
params = {"codes": tuple(airport_codes)}

airport_details = db.tidb_vector_client.execute(search_query, params)
airport_details.get("result")


# Alternatively, we can streamline the process by utilizing a single SQL query to accomplish the search in one step.

# In[19]:


search_query = f"""
    SELECT 
        VEC_Cosine_Distance(se.embedding, :query_vector) as distance, 
        ar.*,
        se.document as airport_review
    FROM 
        airplan_routes ar
    JOIN 
        {TABLE_NAME} se ON ar.airport_code = JSON_UNQUOTE(JSON_EXTRACT(se.meta, '$.airport_code'))
    ORDER BY distance ASC 
    LIMIT 5;
"""
query_vector = embeddings.embed_query(semantic_query)
params = {"query_vector": str(query_vector)}
airport_details = db.tidb_vector_client.execute(search_query, params)
airport_details.get("result")


# In[20]:


# clean up
db.tidb_vector_client.execute("DROP TABLE airplan_routes")


# # Delete
#
# You can remove the TiDB Vector Store by using the `.drop_vectorstore()` method.

# In[21]:


db.drop_vectorstore()
