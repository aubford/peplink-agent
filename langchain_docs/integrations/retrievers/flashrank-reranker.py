#!/usr/bin/env python
# coding: utf-8

# # FlashRank reranker
# 
# >[FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) is the Ultra-lite & Super-fast Python library to add re-ranking to your existing search & retrieval pipelines. It is based on SoTA cross-encoders, with gratitude to all the model owners.
# 
# This notebook shows how to use [flashrank](https://github.com/PrithivirajDamodaran/FlashRank) for document compression and retrieval.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  flashrank')
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  faiss')

# OR  (depending on Python version)

get_ipython().run_line_magic('pip', 'install --upgrade --quiet  faiss_cpu')


# In[2]:


# Helper function for printing docs


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )


# ## Set up the base vector store retriever
# Let's start by initializing a simple vector store retriever and storing the 2023 State of the Union speech (in chunks). We can set up the retriever to retrieve a high number (20) of docs.

# In[3]:


import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()


# In[4]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = TextLoader(
    "../../how_to/state_of_the_union.txt",
).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
for idx, text in enumerate(texts):
    text.metadata["id"] = idx

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
retriever = FAISS.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 20})

query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.invoke(query)
pretty_print_docs(docs)


# ## Doing reranking with FlashRank
# Now let's wrap our base retriever with a `ContextualCompressionRetriever`, using `FlashrankRerank` as a compressor.

# In[5]:


from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
print([doc.metadata["id"] for doc in compressed_docs])


# After reranking, the top 3 documents are different from the top 3 documents retrieved by the base retriever.

# In[6]:


pretty_print_docs(compressed_docs)


# ## QA reranking with FlashRank

# In[7]:


from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever)


# In[8]:


chain.invoke(query)

