#!/usr/bin/env python
# coding: utf-8

# # Infinity Reranker
# 
# `Infinity` is a high-throughput, low-latency REST API for serving text-embeddings, reranking models and clip. 
# For more info, please visit [here](https://github.com/michaelfeil/infinity?tab=readme-ov-file#reranking).
# 
# This notebook shows how to use Infinity Reranker for document compression and retrieval. 

# You can launch an Infinity Server with a reranker model in CLI:
# 
# ```bash
# pip install "infinity-emb[all]"
# infinity_emb v2 --model-id mixedbread-ai/mxbai-rerank-xsmall-v1
# ```

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  infinity_client')


# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  faiss')

# OR  (depending on Python version)

get_ipython().run_line_magic('pip', 'install --upgrade --quiet  faiss-cpu')


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


# ## Reranking with InfinityRerank
# Now let's wrap our base retriever with a `ContextualCompressionRetriever`. We'll use the `InfinityRerank` to rerank the returned results.

# In[5]:


from infinity_client import Client
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.infinity_rerank import InfinityRerank

client = Client(base_url="http://localhost:7997")

compressor = InfinityRerank(client=client, model="mixedbread-ai/mxbai-rerank-xsmall-v1")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)


# In[ ]:




