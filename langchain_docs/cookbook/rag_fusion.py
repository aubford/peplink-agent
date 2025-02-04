#!/usr/bin/env python
# coding: utf-8

# # RAG Fusion
# 
# Re-implemented from [this GitHub repo](https://github.com/Raudaschl/rag-fusion), all credit to original author
# 
# > RAG-Fusion, a search methodology that aims to bridge the gap between traditional search paradigms and the multifaceted dimensions of human queries. Inspired by the capabilities of Retrieval Augmented Generation (RAG), this project goes a step further by employing multiple query generation and Reciprocal Rank Fusion to re-rank search results.

# ## Setup
# 
# For this example, we will use Pinecone and some fake data. To configure Pinecone, set the following environment variable:
# 
# - `PINECONE_API_KEY`: Your Pinecone API key

# In[ ]:


from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


# In[ ]:


all_documents = {
    "doc1": "Climate change and economic impact.",
    "doc2": "Public health concerns due to climate change.",
    "doc3": "Climate change: A social perspective.",
    "doc4": "Technological solutions to climate change.",
    "doc5": "Policy changes needed to combat climate change.",
    "doc6": "Climate change and its impact on biodiversity.",
    "doc7": "Climate change: The science and models.",
    "doc8": "Global warming: A subset of climate change.",
    "doc9": "How climate change affects daily weather.",
    "doc10": "The history of climate change activism.",
}


# In[ ]:


vectorstore = PineconeVectorStore.from_texts(
    list(all_documents.values()), OpenAIEmbeddings(), index_name="rag-fusion"
)


# ## Define the Query Generator
# 
# We will now define a chain to do the query generation

# In[7]:


from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# In[68]:


from langchain import hub

prompt = hub.pull("langchain-ai/rag-fusion-query-generation")


# In[3]:


# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant that generates multiple search queries based on a single input query."),
#     ("user", "Generate multiple search queries related to: {original_query}"),
#     ("user", "OUTPUT (4 queries):")
# ])


# In[5]:


generate_queries = (
    prompt | ChatOpenAI(temperature=0) | StrOutputParser() | (lambda x: x.split("\n"))
)


# ## Define the full chain
# 
# We can now put it all together and define the full chain. This chain:
#     
#     1. Generates a bunch of queries
#     2. Looks up each query in the retriever
#     3. Joins all the results together using reciprocal rank fusion
#     
#     
# Note that it does NOT do a final generation step

# In[50]:


original_query = "impact of climate change"


# In[75]:


vectorstore = PineconeVectorStore.from_existing_index("rag-fusion", OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


# In[76]:


from langchain.load import dumps, loads


def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


# In[77]:


chain = generate_queries | retriever.map() | reciprocal_rank_fusion


# In[78]:


chain.invoke({"original_query": original_query})


# In[ ]:




