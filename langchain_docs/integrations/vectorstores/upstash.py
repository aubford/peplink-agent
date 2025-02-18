#!/usr/bin/env python
# coding: utf-8

# # Upstash Vector
#
# > [Upstash Vector](https://upstash.com/docs/vector/overall/whatisvector) is a serverless vector database designed for working with vector embeddings.
# >
# > The vector langchain integration is a wrapper around the [upstash-vector](https://github.com/upstash/vector-py) package.
# >
# > The python package uses the [vector rest api](https://upstash.com/docs/vector/api/get-started) behind the scenes.

# ## Installation
#
# Create a free vector database from [upstash console](https://console.upstash.com/vector) with the desired dimensions and distance metric.
#
# You can then create an `UpstashVectorStore` instance by:
#
# - Providing the environment variables `UPSTASH_VECTOR_URL` and `UPSTASH_VECTOR_TOKEN`
#
# - Giving them as parameters to the constructor
#
# - Passing an Upstash Vector `Index` instance to the constructor
#
# Also, an `Embeddings` instance is required to turn given texts into embeddings. Here we use `OpenAIEmbeddings` as an example

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install langchain-openai langchain langchain-community upstash-vector"
)


# In[5]:


import os

from langchain_community.vectorstores.upstash import UpstashVectorStore
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_KEY>"
os.environ["UPSTASH_VECTOR_REST_URL"] = "<YOUR_UPSTASH_VECTOR_URL>"
os.environ["UPSTASH_VECTOR_REST_TOKEN"] = "<YOUR_UPSTASH_VECTOR_TOKEN>"

# Create an embeddings instance
embeddings = OpenAIEmbeddings()

# Create a vector store instance
store = UpstashVectorStore(embedding=embeddings)


# An alternative way of creating `UpstashVectorStore` is to [create an Upstash Vector index by selecting a model](https://upstash.com/docs/vector/features/embeddingmodels#using-a-model) and passing `embedding=True`. In this configuration, documents or queries will be sent to Upstash as text and embedded there.
#
# ```python
# store = UpstashVectorStore(embedding=True)
# ```
#
# If you are interested in trying out this approach, you can update the initialization of `store` like above and run the rest of the tutorial.

# ## Load documents
#
# Load an example text file and split it into chunks which can be turned into vector embeddings.

# In[6]:


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

docs[:3]


# ## Inserting documents
#
# The vectorstore embeds text chunks using the embedding object and batch inserts them into the database. This returns an id array of the inserted vectors.

# In[7]:


inserted_vectors = store.add_documents(docs)

inserted_vectors[:5]


# ## Querying
#
# The database can be queried using a vector or a text prompt.
# If a text prompt is used, it's first converted into embedding and then queried.
#
# The `k` parameter specifies how many results to return from the query.

# In[8]:


result = store.similarity_search("technology", k=5)
result


# ## Querying with score
#
# The score of the query can be included for every result.
#
# > The score returned in the query requests is a normalized value between 0 and 1, where 1 indicates the highest similarity and 0 the lowest regardless of the similarity function used. For more information look at the [docs](https://upstash.com/docs/vector/overall/features#vector-similarity-functions).

# In[9]:


result = store.similarity_search_with_score("technology", k=5)

for doc, score in result:
    print(f"{doc.metadata} - {score}")


# ## Namespaces
#
# Namespaces can be used to separate different types of documents. This can increase the efficiency of the queries since the search space is reduced. When no namespace is provided, the default namespace is used.

# In[10]:


store_books = UpstashVectorStore(embedding=embeddings, namespace="books")


# In[11]:


store_books.add_texts(
    [
        "A timeless tale set in the Jazz Age, this novel delves into the lives of affluent socialites, their pursuits of wealth, love, and the elusive American Dream. Amidst extravagant parties and glittering opulence, the story unravels the complexities of desire, ambition, and the consequences of obsession.",
        "Set in a small Southern town during the 1930s, this novel explores themes of racial injustice, moral growth, and empathy through the eyes of a young girl. It follows her father, a principled lawyer, as he defends a black man accused of assaulting a white woman, confronting deep-seated prejudices and challenging societal norms along the way.",
        "A chilling portrayal of a totalitarian regime, this dystopian novel offers a bleak vision of a future world dominated by surveillance, propaganda, and thought control. Through the eyes of a disillusioned protagonist, it explores the dangers of totalitarianism and the erosion of individual freedom in a society ruled by fear and oppression.",
        "Set in the English countryside during the early 19th century, this novel follows the lives of the Bennet sisters as they navigate the intricate social hierarchy of their time. Focusing on themes of marriage, class, and societal expectations, the story offers a witty and insightful commentary on the complexities of romantic relationships and the pursuit of happiness.",
        "Narrated by a disillusioned teenager, this novel follows his journey of self-discovery and rebellion against the phoniness of the adult world. Through a series of encounters and reflections, it explores themes of alienation, identity, and the search for authenticity in a society marked by conformity and hypocrisy.",
        "In a society where emotion is suppressed and individuality is forbidden, one man dares to defy the oppressive regime. Through acts of rebellion and forbidden love, he discovers the power of human connection and the importance of free will.",
        "Set in a future world devastated by environmental collapse, this novel follows a group of survivors as they struggle to survive in a harsh, unforgiving landscape. Amidst scarcity and desperation, they must confront moral dilemmas and question the nature of humanity itself.",
    ],
    [
        {"title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "year": 1925},
        {"title": "To Kill a Mockingbird", "author": "Harper Lee", "year": 1960},
        {"title": "1984", "author": "George Orwell", "year": 1949},
        {"title": "Pride and Prejudice", "author": "Jane Austen", "year": 1813},
        {"title": "The Catcher in the Rye", "author": "J.D. Salinger", "year": 1951},
        {"title": "Brave New World", "author": "Aldous Huxley", "year": 1932},
        {"title": "The Road", "author": "Cormac McCarthy", "year": 2006},
    ],
)


# In[13]:


result = store_books.similarity_search("dystopia", k=3)
result


# ## Metadata Filtering
#
# Metadata can be used to filter the results of a query. You can refer to the [docs](https://upstash.com/docs/vector/features/filtering) to see more complex ways of filtering.

# In[14]:


result = store_books.similarity_search("dystopia", k=3, filter="year < 2000")
result


# ## Getting info about vector database
#
# You can get information about your database like the distance metric dimension using the info function.
#
# > When an insert happens, the database an indexing takes place. While this is happening new vectors can not be queried. `pendingVectorCount` represents the number of vector that are currently being indexed.

# In[15]:


store.info()


# ## Deleting vectors
#
# Vectors can be deleted by their ids

# In[16]:


store.delete(inserted_vectors)


# ## Clearing the vector database
#
# This will clear the vector database

# In[17]:


store.delete(delete_all=True)
store_books.delete(delete_all=True)
