#!/usr/bin/env python
# coding: utf-8

# # Qdrant Sparse Vector
#
# >[Qdrant](https://qdrant.tech/) is an open-source, high-performance vector search engine/database.
#
#
# >`QdrantSparseVectorRetriever` uses [sparse vectors](https://qdrant.tech/articles/sparse-vectors/) introduced in `Qdrant` [v1.7.0](https://qdrant.tech/articles/qdrant-1.7.x/) for document retrieval.
#

# Install the 'qdrant_client' package:

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  qdrant_client")


# In[5]:


from qdrant_client import QdrantClient, models

client = QdrantClient(location=":memory:")
collection_name = "sparse_collection"
vector_name = "sparse_vector"

client.create_collection(
    collection_name,
    vectors_config={},
    sparse_vectors_config={
        vector_name: models.SparseVectorParams(
            index=models.SparseIndexParams(
                on_disk=False,
            )
        )
    },
)


# In[6]:


from langchain_community.retrievers import (
    QdrantSparseVectorRetriever,
)
from langchain_core.documents import Document


# Create a demo encoder function:

# In[7]:


import random


def demo_encoder(_: str) -> tuple[list[int], list[float]]:
    return (
        sorted(random.sample(range(100), 100)),
        [random.uniform(0.1, 1.0) for _ in range(100)],
    )


# Create a retriever with a demo encoder
retriever = QdrantSparseVectorRetriever(
    client=client,
    collection_name=collection_name,
    sparse_vector_name=vector_name,
    sparse_encoder=demo_encoder,
)


# Add some documents:

# In[8]:


docs = [
    Document(
        metadata={
            "title": "Beyond Horizons: AI Chronicles",
            "author": "Dr. Cassandra Mitchell",
        },
        page_content="An in-depth exploration of the fascinating journey of artificial intelligence, narrated by Dr. Mitchell. This captivating account spans the historical roots, current advancements, and speculative futures of AI, offering a gripping narrative that intertwines technology, ethics, and societal implications.",
    ),
    Document(
        metadata={
            "title": "Synergy Nexus: Merging Minds with Machines",
            "author": "Prof. Benjamin S. Anderson",
        },
        page_content="Professor Anderson delves into the synergistic possibilities of human-machine collaboration in 'Synergy Nexus.' The book articulates a vision where humans and AI seamlessly coalesce, creating new dimensions of productivity, creativity, and shared intelligence.",
    ),
    Document(
        metadata={
            "title": "AI Dilemmas: Navigating the Unknown",
            "author": "Dr. Elena Rodriguez",
        },
        page_content="Dr. Rodriguez pens an intriguing narrative in 'AI Dilemmas,' probing the uncharted territories of ethical quandaries arising from AI advancements. The book serves as a compass, guiding readers through the complex terrain of moral decisions confronting developers, policymakers, and society as AI evolves.",
    ),
    Document(
        metadata={
            "title": "Sentient Threads: Weaving AI Consciousness",
            "author": "Prof. Alexander J. Bennett",
        },
        page_content="In 'Sentient Threads,' Professor Bennett unravels the enigma of AI consciousness, presenting a tapestry of arguments that scrutinize the very essence of machine sentience. The book ignites contemplation on the ethical and philosophical dimensions surrounding the quest for true AI awareness.",
    ),
    Document(
        metadata={
            "title": "Silent Alchemy: Unseen AI Alleviations",
            "author": "Dr. Emily Foster",
        },
        page_content="Building upon her previous work, Dr. Foster unveils 'Silent Alchemy,' a profound examination of the covert presence of AI in our daily lives. This illuminating piece reveals the subtle yet impactful ways in which AI invisibly shapes our routines, emphasizing the need for heightened awareness in our technology-driven world.",
    ),
]


# Perform a retrieval:

# In[9]:


retriever.add_documents(docs)


# In[10]:


retriever.invoke(
    "Life and ethical dilemmas of AI",
)
