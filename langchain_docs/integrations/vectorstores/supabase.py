#!/usr/bin/env python
# coding: utf-8

# # Supabase (Postgres)

# >[Supabase](https://supabase.com/docs) is an open-source Firebase alternative. `Supabase` is built on top of `PostgreSQL`, which offers strong SQL querying capabilities and enables a simple interface with already-existing tools and frameworks.
#
# >[PostgreSQL](https://en.wikipedia.org/wiki/PostgreSQL) also known as `Postgres`, is a free and open-source relational database management system (RDBMS) emphasizing extensibility and SQL compliance.
#
# This notebook shows how to use `Supabase` and `pgvector` as your VectorStore.
#
# You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration
#
# To run this notebook, please ensure:
# - the `pgvector` extension is enabled
# - you have installed the `supabase-py` package
# - that you have created a `match_documents` function in your database
# - that you have a `documents` table in your `public` schema similar to the one below.
#
# The following function determines cosine similarity, but you can adjust to your needs.
#
# ```sql
# -- Enable the pgvector extension to work with embedding vectors
# create extension if not exists vector;
#
# -- Create a table to store your documents
# create table
#   documents (
#     id uuid primary key,
#     content text, -- corresponds to Document.pageContent
#     metadata jsonb, -- corresponds to Document.metadata
#     embedding vector (1536) -- 1536 works for OpenAI embeddings, change if needed
#   );
#
# -- Create a function to search for documents
# create function match_documents (
#   query_embedding vector (1536),
#   filter jsonb default '{}'
# ) returns table (
#   id uuid,
#   content text,
#   metadata jsonb,
#   similarity float
# ) language plpgsql as $$
# #variable_conflict use_column
# begin
#   return query
#   select
#     id,
#     content,
#     metadata,
#     1 - (documents.embedding <=> query_embedding) as similarity
#   from documents
#   where metadata @> filter
#   order by documents.embedding <=> query_embedding;
# end;
# $$;
# ```

# In[ ]:


# with pip
get_ipython().run_line_magic("pip", "install --upgrade --quiet  supabase")

# with conda
# !conda install -c conda-forge supabase


# We want to use `OpenAIEmbeddings` so we have to get the OpenAI API Key.

# In[15]:


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")


# In[16]:


if "SUPABASE_URL" not in os.environ:
    os.environ["SUPABASE_URL"] = getpass.getpass("Supabase URL:")


# In[17]:


if "SUPABASE_SERVICE_KEY" not in os.environ:
    os.environ["SUPABASE_SERVICE_KEY"] = getpass.getpass("Supabase Service Key:")


# In[ ]:


# If you're storing your Supabase and OpenAI API keys in a .env file, you can load them with dotenv
from dotenv import load_dotenv

load_dotenv()


# First we'll create a Supabase client and instantiate a OpenAI embeddings class.

# In[19]:


import os

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from supabase.client import Client, create_client

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

embeddings = OpenAIEmbeddings()


# Next we'll load and parse some data for our vector store (skip if you already have documents with embeddings stored in your DB).

# In[20]:


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


# Insert the above documents into the database. Embeddings will automatically be generated for each document. You can adjust the chunk_size based on the amount of documents you have. The default is 500 but lowering it may be necessary.

# In[6]:


vector_store = SupabaseVectorStore.from_documents(
    docs,
    embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
    chunk_size=500,
)


# Alternatively if you already have documents with embeddings in your database, simply instantiate a new `SupabaseVectorStore` directly:

# In[10]:


vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)


# Finally, test it out by performing a similarity search:

# In[ ]:


query = "What did the president say about Ketanji Brown Jackson"
matched_docs = vector_store.similarity_search(query)


# In[ ]:


print(matched_docs[0].page_content)


# ## Similarity search with score
#

# The returned distance score is cosine distance. Therefore, a lower score is better.

# In[9]:


matched_docs = vector_store.similarity_search_with_relevance_scores(query)


# In[10]:


matched_docs[0]


# ## Retriever options
#
# This section goes over different options for how to use SupabaseVectorStore as a retriever.
#
# ### Maximal Marginal Relevance Searches
#
# In addition to using similarity search in the retriever object, you can also use `mmr`.
#

# In[11]:


retriever = vector_store.as_retriever(search_type="mmr")


# In[12]:


matched_docs = retriever.invoke(query)


# In[13]:


for i, d in enumerate(matched_docs):
    print(f"\n## Document {i}\n")
    print(d.page_content)


# In[ ]:
