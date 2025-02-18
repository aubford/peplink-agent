#!/usr/bin/env python
# coding: utf-8

# # Dappier
#
# [Dappier](https://dappier.com) connects any LLM or your Agentic AI to real-time, rights-cleared, proprietary data from trusted sources, making your AI an expert in anything. Our specialized models include Real-Time Web Search, News, Sports, Financial Stock Market Data, Crypto Data, and exclusive content from premium publishers. Explore a wide range of data models in our marketplace at [marketplace.dappier.com](https://marketplace.dappier.com).
#
# [Dappier](https://dappier.com) delivers enriched, prompt-ready, and contextually relevant data strings, optimized for seamless integration with LangChain. Whether you're building conversational AI, recommendation engines, or intelligent search, Dappier's LLM-agnostic RAG models ensure your AI has access to verified, up-to-date data—without the complexity of building and managing your own retrieval pipeline.

# # DappierRetriever
#
# This will help you getting started with the Dappier [retriever](https://python.langchain.com/docs/concepts/retrievers/). For detailed documentation of all DappierRetriever features and configurations head to the [API reference](https://python.langchain.com/en/latest/retrievers/langchain_dappier.retrievers.Dappier.DappierRetriever.html).
#
# ### Setup
#
# Install ``langchain-dappier`` and set environment variable ``DAPPIER_API_KEY``.
#
# ```bash
# pip install -U langchain-dappier
# export DAPPIER_API_KEY="your-api-key"
# ```
#
# We also need to set our Dappier API credentials, which can be generated at the [Dappier site.](https://platform.dappier.com/profile/api-keys).
#
# We can find the supported data models by heading over to the [Dappier marketplace.](https://platform.dappier.com/marketplace)

# If you want to get automated tracing from individual queries, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[ ]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
#
# This retriever lives in the `langchain-dappier` package:

# In[ ]:


get_ipython().run_line_magic("pip", "install -qU langchain-dappier")


# ## Instantiation
#
# - data_model_id: str
#     Data model ID, starting with dm_.
#     You can find the available data model IDs at:
#     [Dappier marketplace.](https://platform.dappier.com/marketplace)
# - k: int
#     Number of documents to return.
# - ref: Optional[str]
#     Site domain where AI recommendations are displayed.
# - num_articles_ref: int
#     Minimum number of articles from the ref domain specified.
#     The rest will come from other sites within the RAG model.
# - search_algorithm: Literal[
#     "most_recent",
#     "most_recent_semantic",
#     "semantic",
#     "trending"
# ]
#     Search algorithm for retrieving articles.
# - api_key: Optional[str]
#     The API key used to interact with the Dappier APIs.

# In[2]:


from langchain_dappier import DappierRetriever

retriever = DappierRetriever(data_model_id="dm_01jagy9nqaeer9hxx8z1sk1jx6")


# ## Usage

# In[3]:


query = "latest tech news"

retriever.invoke(query)


# ## Use within a chain
#
# Like other retrievers, DappierRetriever can be incorporated into LLM applications via [chains](/docs/how_to/sequence/).
#
# We will need a LLM or chat model:

# In[6]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)


# In[7]:


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# In[8]:


chain.invoke(
    "What are the key highlights and outcomes from the latest events covered in the article?"
)


# ## API reference
#
# For detailed documentation of all DappierRetriever features and configurations head to the [API reference](https://python.langchain.com/en/latest/retrievers/langchain_dappier.retrievers.Dappier.DappierRetriever.html).
