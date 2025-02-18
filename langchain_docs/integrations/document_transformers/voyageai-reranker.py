#!/usr/bin/env python
# coding: utf-8

# # VoyageAI Reranker
#
# >[Voyage AI](https://www.voyageai.com/) provides cutting-edge embedding/vectorizations models.
#
# This notebook shows how to use [Voyage AI's rerank endpoint](https://api.voyageai.com/v1/rerank) in a retriever. This builds on top of ideas in the [ContextualCompressionRetriever](/docs/how_to/contextual_compression).

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  voyageai")
get_ipython().run_line_magic("pip", "install --upgrade --quiet  langchain-voyageai")


# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  faiss")

# OR  (depending on Python version)

get_ipython().run_line_magic("pip", "install --upgrade --quiet  faiss-cpu")


# In[13]:


# To obtain your key, create an account on https://www.voyageai.com

import getpass
import os

if "VOYAGE_API_KEY" not in os.environ:
    os.environ["VOYAGE_API_KEY"] = getpass.getpass("Voyage AI API Key:")


# In[14]:


# Helper function for printing docs


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# ## Set up the base vector store retriever
# Let's start by initializing a simple vector store retriever and storing the 2023 State of the Union speech (in chunks). We can set up the retriever to retrieve a high number (20) of docs. You can use any of the following Embeddings models: ([source](https://docs.voyageai.com/docs/embeddings)):
#
# - `voyage-3`
# - `voyage-3-lite`
# - `voyage-large-2`
# - `voyage-code-2`
# - `voyage-2`
# - `voyage-law-2`
# - `voyage-lite-02-instruct`
# - `voyage-finance-2`
# - `voyage-multilingual-2`

# In[15]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings

documents = TextLoader("../../how_to/state_of_the_union.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
retriever = FAISS.from_documents(
    texts, VoyageAIEmbeddings(model="voyage-law-2")
).as_retriever(search_kwargs={"k": 20})

query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.invoke(query)
pretty_print_docs(docs)


# ## Doing reranking with VoyageAIRerank
# Now let's wrap our base retriever with a `ContextualCompressionRetriever`. We'll use the Voyage AI reranker to rerank the returned results. You can use any of the following Reranking models: ([source](https://docs.voyageai.com/docs/reranker)):
#
# - `rerank-2`
# - `rerank-2-lite`
# - `rerank-1`
# - `rerank-lite-1`

# In[16]:


from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import OpenAI
from langchain_voyageai import VoyageAIRerank

llm = OpenAI(temperature=0)
compressor = VoyageAIRerank(
    model="rerank-lite-1", voyageai_api_key=os.environ["VOYAGE_API_KEY"], top_k=3
)
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
    llm=OpenAI(temperature=0), retriever=compression_retriever
)


# In[19]:


chain({"query": query})
