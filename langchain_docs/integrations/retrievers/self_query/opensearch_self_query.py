#!/usr/bin/env python
# coding: utf-8

# # OpenSearch
#
# > [OpenSearch](https://opensearch.org/) is a scalable, flexible, and extensible open-source software suite for search, analytics, and observability applications licensed under Apache 2.0. `OpenSearch` is a distributed search and analytics engine based on `Apache Lucene`.
#
# In this notebook, we'll demo the `SelfQueryRetriever` with an `OpenSearch` vector store.

# ## Creating an OpenSearch vector store
#
# First, we'll want to create an `OpenSearch` vector store and seed it with some data. We've created a small demo set of documents that contain summaries of movies.
#
# **Note:** The self-query retriever requires you to have `lark` installed (`pip install lark`). We also need the `opensearch-py` package.

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  lark opensearch-py")


# In[3]:


import getpass
import os

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

embeddings = OpenAIEmbeddings()


# In[8]:


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
            "rating": 9.9,
            "director": "Andrei Tarkovsky",
            "genre": "science fiction",
        },
    ),
]
vectorstore = OpenSearchVectorSearch.from_documents(
    docs,
    embeddings,
    index_name="opensearch-self-query-demo",
    opensearch_url="http://localhost:9200",
)


# ## Creating our self-querying retriever
# Now we can instantiate our retriever. To do this we'll need to provide some information upfront about the metadata fields that our documents support and a short description of the document contents.

# In[9]:


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

# In[10]:


# This example only specifies a relevant query
retriever.invoke("What are some movies about dinosaurs")


# In[11]:


# This example only specifies a filter
retriever.invoke("I want to watch a movie rated higher than 8.5")


# In[12]:


# This example specifies a query and a filter
retriever.invoke("Has Greta Gerwig directed any movies about women")


# In[13]:


# This example specifies a composite filter
retriever.invoke("What's a highly rated (above 8.5) science fiction film?")


# ## Filter k
#
# We can also use the self query retriever to specify `k`: the number of documents to fetch.
#
# We can do this by passing `enable_limit=True` to the constructor.

# In[14]:


retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    verbose=True,
)


# In[15]:


# This example only specifies a relevant query
retriever.invoke("what are two movies about dinosaurs")


# ## Complex queries in Action!
# We've tried out some simple queries, but what about more complex ones? Let's try out a few more complex queries that utilize the full power of OpenSearch.

# In[16]:


retriever.invoke(
    "what animated or comedy movies have been released in the last 30 years about animated toys?"
)


# In[17]:


vectorstore.client.indices.delete(index="opensearch-self-query-demo")
