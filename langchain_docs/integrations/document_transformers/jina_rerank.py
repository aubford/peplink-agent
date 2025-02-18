#!/usr/bin/env python
# coding: utf-8

# # Jina Reranker

# This notebook shows how to use Jina Reranker for document compression and retrieval.

# In[ ]:


get_ipython().run_line_magic(
    "pip",
    "install -qU langchain langchain-openai langchain-community langchain-text-splitters langchainhub",
)

get_ipython().run_line_magic("pip", "install --upgrade --quiet  faiss")

# OR  (depending on Python version)

get_ipython().run_line_magic("pip", "install --upgrade --quiet  faiss_cpu")


# In[ ]:


# Helper function for printing docs


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# ## Set up the base vector store retriever

# Let's start by initializing a simple vector store retriever and storing the 2023 State of the Union speech (in chunks). We can set up the retriever to retrieve a high number (20) of docs.

# ##### Set the Jina and OpenAI API keys

# In[ ]:


import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()
os.environ["JINA_API_KEY"] = getpass.getpass()


# In[ ]:


from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = TextLoader(
    "../../how_to/state_of_the_union.txt",
).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embedding = JinaEmbeddings(model_name="jina-embeddings-v2-base-en")
retriever = FAISS.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 20})

query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.get_relevant_documents(query)
pretty_print_docs(docs)


# ## Doing reranking with JinaRerank

# Now let's wrap our base retriever with a ContextualCompressionRetriever, using Jina Reranker as a compressor.

# In[ ]:


from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank

compressor = JinaRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.get_relevant_documents(
    "What did the president say about Ketanji Jackson Brown"
)


# In[ ]:


pretty_print_docs(compressed_docs)


# ## QA reranking with Jina Reranker

# In[1]:


from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
retrieval_qa_chat_prompt.pretty_print()


# In[ ]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
chain = create_retrieval_chain(compression_retriever, combine_docs_chain)


# In[ ]:


chain.invoke({"input": query})
