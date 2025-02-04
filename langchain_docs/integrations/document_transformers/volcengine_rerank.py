#!/usr/bin/env python
# coding: utf-8

# # Volcengine Reranker
# 
# This notebook shows how to use Volcengine Reranker for document compression and retrieval. [Volcengine](https://www.volcengine.com/) is a cloud service platform developed by ByteDance, the parent company of TikTok.
# 
# Volcengine's Rerank Service supports reranking up to 50 documents with a maximum of 4000 tokens. For more, please visit [here](https://www.volcengine.com/docs/84313/1254474) and [here](https://www.volcengine.com/docs/84313/1254605).

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  volcengine')


# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  faiss')

# OR  (depending on Python version)

get_ipython().run_line_magic('pip', 'install --upgrade --quiet  faiss-cpu')


# In[ ]:


# To obtain ak/sk: https://www.volcengine.com/docs/84313/1254488

import getpass
import os

if "VOLC_API_AK" not in os.environ:
    os.environ["VOLC_API_AK"] = getpass.getpass("Volcengine API AK:")
if "VOLC_API_SK" not in os.environ:
    os.environ["VOLC_API_SK"] = getpass.getpass("Volcengine API SK:")


# In[3]:


# Helper function for printing docs
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# ## Set up the base vector store retriever
# Let's start by initializing a simple vector store retriever and storing the 2023 State of the Union speech (in chunks). We can set up the retriever to retrieve a high number (20) of docs.

# In[4]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = TextLoader("../../how_to/state_of_the_union.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
retriever = FAISS.from_documents(
    texts, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
).as_retriever(search_kwargs={"k": 20})

query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.invoke(query)
pretty_print_docs(docs)


# ## Reranking with VolcengineRerank
# Now let's wrap our base retriever with a `ContextualCompressionRetriever`. We'll use the `VolcengineRerank` to rerank the returned results.

# In[5]:


from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.volcengine_rerank import VolcengineRerank

compressor = VolcengineRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)


# In[ ]:




