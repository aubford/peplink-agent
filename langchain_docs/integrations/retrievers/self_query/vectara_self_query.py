#!/usr/bin/env python
# coding: utf-8

# # Vectara self-querying 
# 
# [Vectara](https://vectara.com/) is the trusted AI Assistant and Agent platform which focuses on enterprise readiness for mission-critical applications.
# 
# Vectara serverless RAG-as-a-service provides all the components of RAG behind an easy-to-use API, including:
# 1. A way to extract text from files (PDF, PPT, DOCX, etc)
# 2. ML-based chunking that provides state of the art performance.
# 3. The [Boomerang](https://vectara.com/how-boomerang-takes-retrieval-augmented-generation-to-the-next-level-via-grounded-generation/) embeddings model.
# 4. Its own internal vector database where text chunks and embedding vectors are stored.
# 5. A query service that automatically encodes the query into embedding, and retrieves the most relevant text segments, including support for [Hybrid Search](https://docs.vectara.com/docs/api-reference/search-apis/lexical-matching) as well as multiple reranking options such as the [multi-lingual relevance reranker](https://www.vectara.com/blog/deep-dive-into-vectara-multilingual-reranker-v1-state-of-the-art-reranker-across-100-languages), [MMR](https://vectara.com/get-diverse-results-and-comprehensive-summaries-with-vectaras-mmr-reranker/), [UDF reranker](https://www.vectara.com/blog/rag-with-user-defined-functions-based-reranking). 
# 6. An LLM to for creating a [generative summary](https://docs.vectara.com/docs/learn/grounded-generation/grounded-generation-overview), based on the retrieved documents (context), including citations.
# 
# See the [Vectara API documentation](https://docs.vectara.com/docs/) for more information on how to use the API.
# 
# This notebook shows how to use `SelfQueryRetriever` with Vectara.

# # Getting Started
# 
# To get started, use the following steps:
# 1. If you don't already have one, [Sign up](https://www.vectara.com/integrations/langchain) for your free Vectara trial. Once you have completed your sign up you will have a Vectara customer ID. You can find your customer ID by clicking on your name, on the top-right of the Vectara console window.
# 2. Within your account you can create one or more corpora. Each corpus represents an area that stores text data upon ingest from input documents. To create a corpus, use the **"Create Corpus"** button. You then provide a name to your corpus as well as a description. Optionally you can define filtering attributes and apply some advanced options. If you click on your created corpus, you can see its name and corpus ID right on the top.
# 3. Next you'll need to create API keys to access the corpus. Click on the **"Access Control"** tab in the corpus view and then the **"Create API Key"** button. Give your key a name, and choose whether you want query-only or query+index for your key. Click "Create" and you now have an active API key. Keep this key confidential. 
# 
# To use LangChain with Vectara, you'll need to have these three values: `customer ID`, `corpus ID` and `api_key`.
# You can provide those to LangChain in two ways:
# 
# 1. Include in your environment these three variables: `VECTARA_CUSTOMER_ID`, `VECTARA_CORPUS_ID` and `VECTARA_API_KEY`.
# 
#    For example, you can set these variables using os.environ and getpass as follows:
# 
# ```python
# import os
# import getpass
# 
# os.environ["VECTARA_CUSTOMER_ID"] = getpass.getpass("Vectara Customer ID:")
# os.environ["VECTARA_CORPUS_ID"] = getpass.getpass("Vectara Corpus ID:")
# os.environ["VECTARA_API_KEY"] = getpass.getpass("Vectara API Key:")
# ```
# 
# 2. Add them to the `Vectara` vectorstore constructor:
# 
# ```python
# vectara = Vectara(
#                 vectara_customer_id=vectara_customer_id,
#                 vectara_corpus_id=vectara_corpus_id,
#                 vectara_api_key=vectara_api_key
#             )
# ```
# In this notebook we assume they are provided in the environment.
# 
# **Notes:** The self-query retriever requires you to have `lark` installed (`pip install lark`). 

# ## Connecting to Vectara from LangChain
# 
# In this example, we assume that you've created an account and a corpus, and added your `VECTARA_CUSTOMER_ID`, `VECTARA_CORPUS_ID` and `VECTARA_API_KEY` (created with permissions for both indexing and query) as environment variables.
# 
# We further assume the corpus has 4 fields defined as filterable metadata attributes: `year`, `director`, `rating`, and `genre`

# In[1]:


import os

from langchain_core.documents import Document

os.environ["VECTARA_API_KEY"] = "<YOUR_VECTARA_API_KEY>"
os.environ["VECTARA_CORPUS_ID"] = "<YOUR_VECTARA_CORPUS_ID>"
os.environ["VECTARA_CUSTOMER_ID"] = "<YOUR_VECTARA_CUSTOMER_ID>"

from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Vectara
from langchain_openai.chat_models import ChatOpenAI


# ## Dataset
# 
# We first define an example dataset of movie, and upload those to the corpus, along with the metadata:

# In[2]:


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

vectara = Vectara()
for doc in docs:
    vectara.add_texts([doc.page_content], doc_metadata=doc.metadata)


# ## Creating the self-querying retriever
# Now we can instantiate our retriever. To do this we'll need to provide some information upfront about the metadata fields that our documents support and a short description of the document contents.
# 
# We then provide an llm (in this case OpenAI) and the `vectara` vectorstore as arguments:

# In[3]:


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
llm = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=4069)
retriever = SelfQueryRetriever.from_llm(
    llm, vectara, document_content_description, metadata_field_info, verbose=True
)


# ## Self-retrieval Queries
# And now we can try actually using our retriever!

# In[4]:


# This example only specifies a relevant query
retriever.invoke("What are movies about scientists")


# In[5]:


# This example only specifies a filter
retriever.invoke("I want to watch a movie rated higher than 8.5")


# In[6]:


# This example specifies a query and a filter
retriever.invoke("Has Greta Gerwig directed any movies about women")


# In[7]:


# This example specifies a composite filter
retriever.invoke("What's a highly rated (above 8.5) science fiction film?")


# In[8]:


# This example specifies a query and composite filter
retriever.invoke(
    "What's a movie after 1990 but before 2005 that's all about toys, and preferably is animated"
)


# ## Filter k
# 
# We can also use the self query retriever to specify `k`: the number of documents to fetch.
# 
# We can do this by passing `enable_limit=True` to the constructor.

# In[9]:


retriever = SelfQueryRetriever.from_llm(
    llm,
    vectara,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    verbose=True,
)


# This is cool, we can include the number of results we would like to see in the query and the self retriever would correctly understand it. For example, let's look for 

# In[10]:


# This example only specifies a relevant query
retriever.invoke("what are two movies with a rating above 8.5")


# In[ ]:




