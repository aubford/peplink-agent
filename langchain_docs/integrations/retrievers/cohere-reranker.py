#!/usr/bin/env python
# coding: utf-8

# # Cohere reranker
#
# >[Cohere](https://cohere.ai/about) is a Canadian startup that provides natural language processing models that help companies improve human-machine interactions.
#
# This notebook shows how to use [Cohere's rerank endpoint](https://docs.cohere.com/docs/reranking) in a retriever. This builds on top of ideas in the [ContextualCompressionRetriever](/docs/how_to/contextual_compression).

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  cohere")


# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  faiss")

# OR  (depending on Python version)

get_ipython().run_line_magic("pip", "install --upgrade --quiet  faiss-cpu")


# In[13]:


# get a new token: https://dashboard.cohere.ai/

import getpass
import os

if "COHERE_API_KEY" not in os.environ:
    os.environ["COHERE_API_KEY"] = getpass.getpass("Cohere API Key:")


# In[14]:


# Helper function for printing docs


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# ## Set up the base vector store retriever
# Let's start by initializing a simple vector store retriever and storing the 2023 State of the Union speech (in chunks). We can set up the retriever to retrieve a high number (20) of docs.

# In[15]:


from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = TextLoader("../../how_to/state_of_the_union.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
retriever = FAISS.from_documents(
    texts, CohereEmbeddings(model="embed-english-v3.0")
).as_retriever(search_kwargs={"k": 20})

query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.invoke(query)
pretty_print_docs(docs)


# ## Doing reranking with CohereRerank
# Now let's wrap our base retriever with a `ContextualCompressionRetriever`. We'll add an `CohereRerank`, uses the Cohere rerank endpoint to rerank the returned results.
# Do note that it is mandatory to specify the model name in CohereRerank!

# In[ ]:


from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere

llm = Cohere(temperature=0)
compressor = CohereRerank(model="rerank-english-v3.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)


# You can of course use this retriever within a QA pipeline

# In[17]:


from langchain.chains import RetrievalQA


# In[18]:


chain = RetrievalQA.from_chain_type(
    llm=Cohere(temperature=0), retriever=compression_retriever
)


# In[19]:


chain({"query": query})
