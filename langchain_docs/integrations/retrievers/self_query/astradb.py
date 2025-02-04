#!/usr/bin/env python
# coding: utf-8

# # Astra DB (Cassandra)
# 
# >[DataStax Astra DB](https://docs.datastax.com/en/astra/home/astra.html) is a serverless vector-capable database built on `Cassandra` and made conveniently available through an easy-to-use JSON API.
# 
# In the walkthrough, we'll demo the `SelfQueryRetriever` with an `Astra DB` vector store.

# ## Creating an Astra DB vector store
# First we'll want to create an Astra DB VectorStore and seed it with some data. We've created a small demo set of documents that contain summaries of movies.
# 
# NOTE: The self-query retriever requires you to have `lark` installed (`pip install lark`). We also need the `astrapy` package.

# In[5]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet lark astrapy langchain-openai')


# We want to use `OpenAIEmbeddings` so we have to get the OpenAI API Key.

# In[ ]:


import os
from getpass import getpass

from langchain_openai.embeddings import OpenAIEmbeddings

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("OpenAI API Key:")

embeddings = OpenAIEmbeddings()


# Create the Astra DB VectorStore:
# 
# - the API Endpoint looks like `https://01234567-89ab-cdef-0123-456789abcdef-us-east1.apps.astra.datastax.com`
# - the Token looks like `AstraCS:6gBhNmsk135....`

# In[ ]:


ASTRA_DB_API_ENDPOINT = input("ASTRA_DB_API_ENDPOINT = ")
ASTRA_DB_APPLICATION_TOKEN = getpass("ASTRA_DB_APPLICATION_TOKEN = ")


# In[ ]:


from langchain_community.vectorstores import AstraDB
from langchain_core.documents import Document

docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "science fiction",
            "rating": 9.9,
        },
    ),
]

vectorstore = AstraDB.from_documents(
    docs,
    embeddings,
    collection_name="astra_self_query_demo",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)


# ## Creating our self-querying retriever
# Now we can instantiate our retriever. To do this we'll need to provide some information upfront about the metadata fields that our documents support and a short description of the document contents.

# In[ ]:


from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import OpenAI

metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]
document_content_description = "Brief summary of a movie"
llm = OpenAI(temperature=0)

retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)


# ## Testing it out
# And now we can try actually using our retriever!

# In[ ]:


# This example only specifies a relevant query
retriever.invoke("What are some movies about dinosaurs?")


# In[ ]:


# This example specifies a filter
retriever.invoke("I want to watch a movie rated higher than 8.5")


# In[ ]:


# This example only specifies a query and a filter
retriever.invoke("Has Greta Gerwig directed any movies about women")


# In[ ]:


# This example specifies a composite filter
retriever.invoke("What's a highly rated (above 8.5), science fiction movie ?")


# In[ ]:


# This example specifies a query and composite filter
retriever.invoke(
    "What's a movie about toys after 1990 but before 2005, and is animated"
)


# ## Filter k
# 
# We can also use the self query retriever to specify `k`: the number of documents to fetch.
# 
# We can do this by passing `enable_limit=True` to the constructor.

# In[ ]:


retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
    enable_limit=True,
)


# In[ ]:


# This example only specifies a relevant query
retriever.invoke("What are two movies about dinosaurs?")


# ## Cleanup
# 
# If you want to completely delete the collection from your Astra DB instance, run this.
# 
# _(You will lose the data you stored in it.)_

# In[ ]:


vectorstore.delete_collection()

