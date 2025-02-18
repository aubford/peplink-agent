#!/usr/bin/env python
# coding: utf-8

# # AskNews
#
# > [AskNews](https://asknews.app) infuses any LLM with the latest global news (or historical news), using a single natural language query. Specifically, AskNews is enriching over 300k articles per day by translating, summarizing, extracting entities, and indexing them into hot and cold vector databases. AskNews puts these vector databases on a low-latency endpoint for you. When you query AskNews, you get back a prompt-optimized string that contains all the most pertinent enrichments (e.g. entities, classifications, translation, summarization). This means that you do not need to manage your own news RAG, and you do not need to worry about how to properly convey news information in a condensed way to your LLM.
# > AskNews is also committed to transparency, which is why our coverage is monitored and diversified across hundreds of countries, 13 languages, and 50 thousand sources. If you'd like to track our source coverage, you can visit our [transparency dashboard](https://asknews.app/en/transparency).
#
# ## Setup
#
# The integration lives in the `langchain-community` package. We also need to install the `asknews` package itself.
#
# ```bash
# pip install -U langchain-community asknews
# ```
#
# We also need to set our AskNews API credentials, which can be generated at the [AskNews console](https://my.asknews.app).

# In[1]:


import getpass
import os

os.environ["ASKNEWS_CLIENT_ID"] = getpass.getpass()
os.environ["ASKNEWS_CLIENT_SECRET"] = getpass.getpass()


# It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability

# In[ ]:


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()


# ## Usage

# In[13]:


from langchain_community.retrievers import AskNewsRetriever

retriever = AskNewsRetriever(k=3)

retriever.invoke("impact of fed policy on the tech sector")


# In[19]:


# you have full control on filtering by category, time, pagination, and even the search method you use.
from datetime import datetime, timedelta

start = (datetime.now() - timedelta(days=7)).timestamp()
end = datetime.now().timestamp()

retriever = AskNewsRetriever(
    k=3,
    categories=["Business", "Technology"],
    start_timestamp=int(start),  # defaults to 48 hours ago
    end_timestamp=int(end),  # defaults to now
    method="kw",  # defaults to "nl", natural language, can also be "kw" for keyword search
    offset=10,  # allows you to paginate results
)

retriever.invoke("federal reserve S&P500")


# ## Chaining
#
# We can easily combine this retriever in to a chain.

# In[3]:


import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()


# In[11]:


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template(
    """The following news articles may come in handy for answering the question:

{context}

Question:

{question}"""
)
chain = (
    RunnablePassthrough.assign(context=(lambda x: x["question"]) | retriever)
    | prompt
    | ChatOpenAI(model="gpt-4-1106-preview")
    | StrOutputParser()
)


# In[12]:


chain.invoke({"question": "What is the impact of fed policy on the tech sector?"})


# In[ ]:
