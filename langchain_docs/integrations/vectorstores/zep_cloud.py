#!/usr/bin/env python
# coding: utf-8

# # Zep Cloud
# > Recall, understand, and extract data from chat histories. Power personalized AI experiences.
#
# > [Zep](https://www.getzep.com) is a long-term memory service for AI Assistant apps.
# > With Zep, you can provide AI assistants with the ability to recall past conversations, no matter how distant,
# > while also reducing hallucinations, latency, and cost.
#
# > See [Zep Cloud Installation Guide](https://help.getzep.com/sdks)
#
# ## Usage
#
# In the examples below, we're using Zep's auto-embedding feature which automatically embeds documents on the Zep server
# using low-latency embedding models.
#
# ## Note
# - These examples use Zep's async interfaces. Call sync interfaces by removing the `a` prefix from the method names.

# ## Load or create a Collection from documents

# In[1]:


from uuid import uuid4

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import ZepCloudVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

ZEP_API_KEY = "<your zep project key>"  # You can generate your zep project key from the Zep dashboard
collection_name = f"babbage{uuid4().hex}"  # a unique collection name. alphanum only

# load the document
article_url = "https://www.gutenberg.org/cache/epub/71292/pg71292.txt"
loader = WebBaseLoader(article_url)
documents = loader.load()

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Instantiate the VectorStore. Since the collection does not already exist in Zep,
# it will be created and populated with the documents we pass in.
vs = ZepCloudVectorStore.from_documents(
    docs,
    embedding=None,
    collection_name=collection_name,
    api_key=ZEP_API_KEY,
)


# In[2]:


# wait for the collection embedding to complete


async def wait_for_ready(collection_name: str) -> None:
    import time

    from zep_cloud.client import AsyncZep

    client = AsyncZep(api_key=ZEP_API_KEY)

    while True:
        c = await client.document.get_collection(collection_name)
        print(
            "Embedding status: "
            f"{c.document_embedded_count}/{c.document_count} documents embedded"
        )
        time.sleep(1)
        if c.document_embedded_count == c.document_count:
            break


await wait_for_ready(collection_name)


# ## Simarility Search Query over the Collection

# In[4]:


# query it
query = "what is the structure of our solar system?"
docs_scores = await vs.asimilarity_search_with_relevance_scores(query, k=3)

# print results
for d, s in docs_scores:
    print(d.page_content, " -> ", s, "\n====\n")


# ## Search over Collection Re-ranked by MMR
#
# Zep offers native, hardware-accelerated MMR re-ranking of search results.

# In[5]:


query = "what is the structure of our solar system?"
docs = await vs.asearch(query, search_type="mmr", k=3)

for d in docs:
    print(d.page_content, "\n====\n")


# # Filter by Metadata
#
# Use a metadata filter to narrow down results. First, load another book: "Adventures of Sherlock Holmes"

# In[3]:


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

# In[4]:


query = "Was he interested in astronomy?"
docs = await vs.asearch(query, search_type="similarity", k=3)

for d in docs:
    print(d.page_content, " -> ", d.metadata, "\n====\n")


# Now, we set up a filter

# In[5]:


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
