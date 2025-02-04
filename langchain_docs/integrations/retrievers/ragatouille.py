#!/usr/bin/env python
# coding: utf-8

# # RAGatouille
# 
# 
# >[RAGatouille](https://github.com/bclavie/RAGatouille) makes it as simple as can be to use `ColBERT`!
# >
# >[ColBERT](https://github.com/stanford-futuredata/ColBERT) is a fast and accurate retrieval model, enabling scalable BERT-based search over large text collections in tens of milliseconds.
# >
# >See the [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) paper.
# 
# We can use this as a [retriever](/docs/how_to#retrievers). It will show functionality specific to this integration. After going through, it may be useful to explore [relevant use-case pages](/docs/how_to#qa-with-rag) to learn how to use this vector store as part of a larger chain.
# 
# This page covers how to use [RAGatouille](https://github.com/bclavie/RAGatouille) as a retriever in a LangChain chain. 
# 
# ## Setup
# 
# The integration lives in the `ragatouille` package.
# 
# ```bash
# pip install -U ragatouille
# ```

# ## Usage
# 
# This example is taken from their documentation

# In[2]:


from ragatouille import RAGPretrainedModel

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")


# In[3]:


import requests


def get_wikipedia_page(title: str):
    """
    Retrieve the full text content of a Wikipedia page.

    :param title: str - Title of the Wikipedia page.
    :return: str - Full text content of the page as raw string.
    """
    # Wikipedia API endpoint
    URL = "https://en.wikipedia.org/w/api.php"

    # Parameters for the API request
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }

    # Custom User-Agent header to comply with Wikipedia's best practices
    headers = {"User-Agent": "RAGatouille_tutorial/0.0.1 (ben@clavie.eu)"}

    response = requests.get(URL, params=params, headers=headers)
    data = response.json()

    # Extracting page content
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None


# In[4]:


full_document = get_wikipedia_page("Hayao_Miyazaki")


# In[5]:


RAG.index(
    collection=[full_document],
    index_name="Miyazaki-123",
    max_document_length=180,
    split_documents=True,
)


# In[6]:


results = RAG.search(query="What animation studio did Miyazaki found?", k=3)


# In[7]:


results


# We can then convert easily to a LangChain retriever! We can pass in any kwargs we want when creating (like `k`)

# In[8]:


retriever = RAG.as_langchain_retriever(k=3)


# In[10]:


retriever.invoke("What animation studio did Miyazaki found?")


# ## Chaining
# 
# We can easily combine this retriever in to a chain.

# In[11]:


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}"""
)

llm = ChatOpenAI()

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# In[12]:


retrieval_chain.invoke({"input": "What animation studio did Miyazaki found?"})


# In[13]:


for s in retrieval_chain.stream({"input": "What animation studio did Miyazaki found?"}):
    print(s.get("answer", ""), end="")


# In[ ]:




