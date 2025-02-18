#!/usr/bin/env python
# coding: utf-8

# # Neo4j Vector Index
#
# >[Neo4j](https://neo4j.com/) is an open-source graph database with integrated support for vector similarity search
#
# It supports:
#
# - approximate nearest neighbor search
# - Euclidean similarity and cosine similarity
# - Hybrid search combining vector and keyword searches
#
# This notebook shows how to use the Neo4j vector index (`Neo4jVector`).

# See the [installation instruction](https://neo4j.com/docs/operations-manual/current/installation/).

# In[ ]:


# Pip install necessary package
get_ipython().run_line_magic("pip", "install --upgrade --quiet  neo4j")
get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet  langchain-openai langchain-neo4j"
)
get_ipython().run_line_magic("pip", "install --upgrade --quiet  tiktoken")


# We want to use `OpenAIEmbeddings` so we have to get the OpenAI API Key.

# In[2]:


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")


# In[3]:


from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


# In[4]:


loader = TextLoader("../../how_to/state_of_the_union.txt")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


# In[5]:


# Neo4jVector requires the Neo4j database credentials

url = "bolt://localhost:7687"
username = "neo4j"
password = "password"

# You can also use environment variables instead of directly passing named parameters
# os.environ["NEO4J_URI"] = "bolt://localhost:7687"
# os.environ["NEO4J_USERNAME"] = "neo4j"
# os.environ["NEO4J_PASSWORD"] = "pleaseletmein"


# ## Similarity Search with Cosine Distance (Default)

# In[6]:


# The Neo4jVector Module will connect to Neo4j and create a vector index if needed.

db = Neo4jVector.from_documents(
    docs, OpenAIEmbeddings(), url=url, username=username, password=password
)


# In[7]:


query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = db.similarity_search_with_score(query, k=2)


# In[8]:


for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)


# ## Working with vectorstore
#
# Above, we created a vectorstore from scratch. However, often times we want to work with an existing vectorstore.
# In order to do that, we can initialize it directly.

# In[9]:


index_name = "vector"  # default index name

store = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name=index_name,
)


# We can also initialize a vectorstore from existing graph using the `from_existing_graph` method. This method pulls relevant text information from the database, and calculates and stores the text embeddings back to the database.

# In[10]:


# First we create sample data in graph
store.query(
    "CREATE (p:Person {name: 'Tomaz', location:'Slovenia', hobby:'Bicycle', age: 33})"
)


# In[11]:


# Now we initialize from existing graph
existing_graph = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name="person_index",
    node_label="Person",
    text_node_properties=["name", "location"],
    embedding_node_property="embedding",
)
result = existing_graph.similarity_search("Slovenia", k=1)


# In[12]:


result[0]


# Neo4j also supports relationship vector indexes, where an embedding is stored as a relationship property and indexed. A relationship vector index cannot be populated via LangChain, but you can connect it to existing relationship vector indexes.

# In[13]:


# First we create sample data and index in graph
store.query(
    "MERGE (p:Person {name: 'Tomaz'}) "
    "MERGE (p1:Person {name:'Leann'}) "
    "MERGE (p1)-[:FRIEND {text:'example text', embedding:$embedding}]->(p2)",
    params={"embedding": OpenAIEmbeddings().embed_query("example text")},
)
# Create a vector index
relationship_index = "relationship_vector"
store.query(
    """
CREATE VECTOR INDEX $relationship_index
IF NOT EXISTS
FOR ()-[r:FRIEND]-() ON (r.embedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}}
""",
    params={"relationship_index": relationship_index},
)


# In[14]:


relationship_vector = Neo4jVector.from_existing_relationship_index(
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name=relationship_index,
    text_node_property="text",
)
relationship_vector.similarity_search("Example")


# ### Metadata filtering
#
# Neo4j vector store also supports metadata filtering by combining parallel runtime and exact nearest neighbor search.
# _Requires Neo4j 5.18 or greater version._
#
# Equality filtering has the following syntax.

# In[15]:


existing_graph.similarity_search(
    "Slovenia",
    filter={"hobby": "Bicycle", "name": "Tomaz"},
)


# Metadata filtering also support the following operators:
#
# * `$eq: Equal`
# * `$ne: Not Equal`
# * `$lt: Less than`
# * `$lte: Less than or equal`
# * `$gt: Greater than`
# * `$gte: Greater than or equal`
# * `$in: In a list of values`
# * `$nin: Not in a list of values`
# * `$between: Between two values`
# * `$like: Text contains value`
# * `$ilike: lowered text contains value`

# In[16]:


existing_graph.similarity_search(
    "Slovenia",
    filter={"hobby": {"$eq": "Bicycle"}, "age": {"$gt": 15}},
)


# You can also use `OR` operator between filters

# In[17]:


existing_graph.similarity_search(
    "Slovenia",
    filter={"$or": [{"hobby": {"$eq": "Bicycle"}}, {"age": {"$gt": 15}}]},
)


# ### Add documents
# We can add documents to the existing vectorstore.

# In[18]:


store.add_documents([Document(page_content="foo")])


# In[19]:


docs_with_score = store.similarity_search_with_score("foo")


# In[20]:


docs_with_score[0]


# ## Customize response with retrieval query
#
# You can also customize responses by using a custom Cypher snippet that can fetch other information from the graph.
# Under the hood, the final Cypher statement is constructed like so:
#
# ```
# read_query = (
#   "CALL db.index.vector.queryNodes($index, $k, $embedding) "
#   "YIELD node, score "
# ) + retrieval_query
# ```
#
# The retrieval query must return the following three columns:
#
# * `text`: Union[str, Dict] = Value used to populate `page_content` of a document
# * `score`: Float = Similarity score
# * `metadata`: Dict = Additional metadata of a document
#
# Learn more in this [blog post](https://medium.com/neo4j/implementing-rag-how-to-write-a-graph-retrieval-query-in-langchain-74abf13044f2).

# In[21]:


retrieval_query = """
RETURN "Name:" + node.name AS text, score, {foo:"bar"} AS metadata
"""
retrieval_example = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name="person_index",
    retrieval_query=retrieval_query,
)
retrieval_example.similarity_search("Foo", k=1)


# Here is an example of passing all node properties except for `embedding` as a dictionary to `text` column,

# In[22]:


retrieval_query = """
RETURN node {.name, .age, .hobby} AS text, score, {foo:"bar"} AS metadata
"""
retrieval_example = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name="person_index",
    retrieval_query=retrieval_query,
)
retrieval_example.similarity_search("Foo", k=1)


# You can also pass Cypher parameters to the retrieval query.
# Parameters can be used for additional filtering, traversals, etc...

# In[23]:


retrieval_query = """
RETURN node {.*, embedding:Null, extra: $extra} AS text, score, {foo:"bar"} AS metadata
"""
retrieval_example = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name="person_index",
    retrieval_query=retrieval_query,
)
retrieval_example.similarity_search("Foo", k=1, params={"extra": "ParamInfo"})


# ## Hybrid search (vector + keyword)
#
# Neo4j integrates both vector and keyword indexes, which allows you to use a hybrid search approach

# In[24]:


# The Neo4jVector Module will connect to Neo4j and create a vector and keyword indices if needed.
hybrid_db = Neo4jVector.from_documents(
    docs,
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    search_type="hybrid",
)


# To load the hybrid search from existing indexes, you have to provide both the vector and keyword indices

# In[25]:


index_name = "vector"  # default index name
keyword_index_name = "keyword"  # default keyword index name

store = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name=index_name,
    keyword_index_name=keyword_index_name,
    search_type="hybrid",
)


# ## Retriever options
#
# This section shows how to use `Neo4jVector` as a retriever.

# In[26]:


retriever = store.as_retriever()
retriever.invoke(query)[0]


# ## Question Answering with Sources
#
# This section goes over how to do question-answering with sources over an Index. It does this by using the `RetrievalQAWithSourcesChain`, which does the lookup of the documents from an Index.

# In[27]:


from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI


# In[28]:


chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0), chain_type="stuff", retriever=retriever
)


# In[29]:


chain.invoke(
    {"question": "What did the president say about Justice Breyer"},
    return_only_outputs=True,
)


# In[ ]:
