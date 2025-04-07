#!/usr/bin/env python
# coding: utf-8

# # Zep
# > Recall, understand, and extract data from chat histories. Power personalized AI experiences.
# 
# > [Zep](https://www.getzep.com) is a long-term memory service for AI Assistant apps.
# > With Zep, you can provide AI assistants with the ability to recall past conversations, no matter how distant,
# > while also reducing hallucinations, latency, and cost.
# 
# > Interested in Zep Cloud? See [Zep Cloud Installation Guide](https://help.getzep.com/sdks) and [Zep Cloud Vector Store example](https://help.getzep.com/langchain/examples/vectorstore-example)
# 
# ## Open Source Installation and Setup
# 
# > Zep Open Source project: [https://github.com/getzep/zep](https://github.com/getzep/zep)
# >
# > Zep Open Source Docs: [https://docs.getzep.com/](https://docs.getzep.com/)
# 
# You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration
# 
# ## Usage
# 
# In the examples below, we're using Zep's auto-embedding feature which automatically embeds documents on the Zep server 
# using low-latency embedding models.
# 
# ## Note
# - These examples use Zep's async interfaces. Call sync interfaces by removing the `a` prefix from the method names.
# - If you pass in an `Embeddings` instance Zep will use this to embed documents rather than auto-embed them.
# You must also set your document collection to `isAutoEmbedded === false`. 
# - If you set your collection to `isAutoEmbedded === false`, you must pass in an `Embeddings` instance.

# ## Load or create a Collection from documents

# In[1]:


from uuid import uuid4

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import ZepVectorStore
from langchain_community.vectorstores.zep import CollectionConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter

ZEP_API_URL = "http://localhost:8000"  # this is the API url of your Zep instance
ZEP_API_KEY = "<optional_key>"  # optional API Key for your Zep instance
collection_name = f"babbage{uuid4().hex}"  # a unique collection name. alphanum only

# Collection config is needed if we're creating a new Zep Collection
config = CollectionConfig(
    name=collection_name,
    description="<optional description>",
    metadata={"optional_metadata": "associated with the collection"},
    is_auto_embedded=True,  # we'll have Zep embed our documents using its low-latency embedder
    embedding_dimensions=1536,  # this should match the model you've configured Zep to use.
)

# load the document
article_url = "https://www.gutenberg.org/cache/epub/71292/pg71292.txt"
loader = WebBaseLoader(article_url)
documents = loader.load()

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Instantiate the VectorStore. Since the collection does not already exist in Zep,
# it will be created and populated with the documents we pass in.
vs = ZepVectorStore.from_documents(
    docs,
    collection_name=collection_name,
    config=config,
    api_url=ZEP_API_URL,
    api_key=ZEP_API_KEY,
    embedding=None,  # we'll have Zep embed our documents using its low-latency embedder
)


# In[2]:


# wait for the collection embedding to complete


async def wait_for_ready(collection_name: str) -> None:
    import time

    from zep_python import ZepClient

    client = ZepClient(ZEP_API_URL, ZEP_API_KEY)

    while True:
        c = await client.document.aget_collection(collection_name)
        print(
            "Embedding status: "
            f"{c.document_embedded_count}/{c.document_count} documents embedded"
        )
        time.sleep(1)
        if c.status == "ready":
            break


await wait_for_ready(collection_name)


# ## Simarility Search Query over the Collection

# In[3]:


# query it
query = "what is the structure of our solar system?"
docs_scores = await vs.asimilarity_search_with_relevance_scores(query, k=3)

# print results
for d, s in docs_scores:
    print(d.page_content, " -> ", s, "\n====\n")


# ## Search over Collection Re-ranked by MMR
# 
# Zep offers native, hardware-accelerated MMR re-ranking of search results.

# In[4]:


query = "what is the structure of our solar system?"
docs = await vs.asearch(query, search_type="mmr", k=3)

for d in docs:
    print(d.page_content, "\n====\n")


# # Filter by Metadata
# 
# Use a metadata filter to narrow down results. First, load another book: "Adventures of Sherlock Holmes"

# In[5]:


# Let's add more content to the existing Collection
article_url = "https://www.gutenberg.org/files/48320/48320-0.txt"
loader = WebBaseLoader(article_url)
documents = loader.load()

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

await vs.aadd_documents(docs)

await wait_for_ready(collection_name)


# We see results from both books. Note the `source` metadata

# In[6]:


query = "Was he interested in astronomy?"
docs = await vs.asearch(query, search_type="similarity", k=3)

for d in docs:
    print(d.page_content, " -> ", d.metadata, "\n====\n")


# Now, we set up a filter

# In[7]:


filter = {
    "where": {
        "jsonpath": (
            "$[*] ? (@.source == 'https://www.gutenberg.org/files/48320/48320-0.txt')"
        )
    },
}

docs = await vs.asearch(query, search_type="similarity", metadata=filter, k=3)

for d in docs:
    print(d.page_content, " -> ", d.metadata, "\n====\n")


# In[ ]:




