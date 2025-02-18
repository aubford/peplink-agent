#!/usr/bin/env python
# coding: utf-8

# # LLMLingua Document Compressor
#
# >[LLMLingua](https://github.com/microsoft/LLMLingua) utilizes a compact, well-trained language model (e.g., GPT2-small, LLaMA-7B) to identify and remove non-essential tokens in prompts. This approach enables efficient inference with large language models (LLMs), achieving up to 20x compression with minimal performance loss.
#
# This notebook shows how to use LLMLingua as a document compressor.

# In[5]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  llmlingua accelerate")


# In[1]:


# Helper function for printing docs


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# ## Set up the base vector store retriever
# Let's start by initializing a simple vector store retriever and storing the 2023 State of the Union speech (in chunks). We can set up the retriever to retrieve a high number (20) of docs.

# In[2]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = TextLoader(
    "../../how_to/state_of_the_union.txt",
).load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
retriever = FAISS.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 20})

query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.invoke(query)
pretty_print_docs(docs)


# ## Doing compression with LLMLingua
# Now let’s wrap our base retriever with a `ContextualCompressionRetriever`, using `LLMLinguaCompressor` as a compressor.

# In[6]:


from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import LLMLinguaCompressor
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

compressor = LLMLinguaCompressor(model_name="openai-community/gpt2", device_map="cpu")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)


# ## QA generation with LLMLingua
#
# We can see what it looks like to use this in the generation step now

# In[7]:


from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever)


# In[8]:


chain.invoke({"query": query})


# In[ ]:
