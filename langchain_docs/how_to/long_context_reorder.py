#!/usr/bin/env python
# coding: utf-8

# # How to reorder retrieved results to mitigate the "lost in the middle" effect
#
# Substantial performance degradations in [RAG](/docs/tutorials/rag) applications have been [documented](https://arxiv.org/abs/2307.03172) as the number of retrieved documents grows (e.g., beyond ten). In brief: models are liable to miss relevant information in the middle of long contexts.
#
# By contrast, queries against vector stores will typically return documents in descending order of relevance (e.g., as measured by cosine similarity of [embeddings](/docs/concepts/embedding_models)).
#
# To mitigate the ["lost in the middle"](https://arxiv.org/abs/2307.03172) effect, you can re-order documents after retrieval such that the most relevant documents are positioned at extrema (e.g., the first and last pieces of context), and the least relevant documents are positioned in the middle. In some cases this can help surface the most relevant information to LLMs.
#
# The [LongContextReorder](https://python.langchain.com/api_reference/community/document_transformers/langchain_community.document_transformers.long_context_reorder.LongContextReorder.html) document transformer implements this re-ordering procedure. Below we demonstrate an example.

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install -qU langchain langchain-community langchain-openai"
)


# First we embed some artificial documents and index them in a basic in-memory vector store. We will use [OpenAI](/docs/integrations/providers/openai/) embeddings, but any LangChain vector store or embeddings model will suffice.

# In[1]:


from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# Get embeddings.
embeddings = OpenAIEmbeddings()

texts = [
    "Basquetball is a great sport.",
    "Fly me to the moon is one of my favourite songs.",
    "The Celtics are my favourite team.",
    "This is a document about the Boston Celtics",
    "I simply love going to the movies",
    "The Boston Celtics won the game by 20 points",
    "This is just a random text.",
    "Elden Ring is one of the best games in the last 15 years.",
    "L. Kornet is one of the best Celtics players.",
    "Larry Bird was an iconic NBA player.",
]

# Create a retriever
retriever = InMemoryVectorStore.from_texts(texts, embedding=embeddings).as_retriever(
    search_kwargs={"k": 10}
)
query = "What can you tell me about the Celtics?"

# Get relevant documents ordered by relevance score
docs = retriever.invoke(query)
for doc in docs:
    print(f"- {doc.page_content}")


# Note that documents are returned in descending order of relevance to the query. The `LongContextReorder` document transformer will implement the re-ordering described above:

# In[2]:


from langchain_community.document_transformers import LongContextReorder

# Reorder the documents:
# Less relevant document will be at the middle of the list and more
# relevant elements at beginning / end.
reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)

# Confirm that the 4 relevant documents are at beginning and end.
for doc in reordered_docs:
    print(f"- {doc.page_content}")


# Below, we show how to incorporate the re-ordered documents into a simple question-answering chain:

# In[3]:


from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

prompt_template = """
Given these texts:
-----
{context}
-----
Please answer the following question:
{query}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "query"],
)

# Create and invoke the chain:
chain = create_stuff_documents_chain(llm, prompt)
response = chain.invoke({"context": reordered_docs, "query": query})
print(response)
