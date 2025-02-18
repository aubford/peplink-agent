#!/usr/bin/env python
# coding: utf-8

# Press Releases Data
# =
#
# Press Releases data powered by [Kay.ai](https://kay.ai).
#
# >Press releases are used by companies to announce something noteworthy, including product launches, financial performance reports, partnerships, and other significant news. They are widely used by analysts to track corporate strategy, operational updates and financial performance.
# Kay.ai obtains press releases of all US public companies from a variety of sources, which include the company's official press room and partnerships with various data API providers.
# This data is updated till Sept 30th for free access, if you want to access the real-time feed, reach out to us at hello@kay.ai or [tweet at us](https://twitter.com/vishalrohra_)

# Setup
# =
#
# First you will need to install the `kay` package. You will also need an API key: you can get one for free at [https://kay.ai](https://kay.ai/). Once you have an API key, you must set it as an environment variable `KAY_API_KEY`.
#
# In this example we're going to use the `KayAiRetriever`. Take a look at the [kay notebook](/docs/integrations/retrievers/kay) for more detailed information for the parmeters that it accepts.

# Examples
# =

# In[1]:


# Setup API keys for Kay and OpenAI
from getpass import getpass

KAY_API_KEY = getpass()
OPENAI_API_KEY = getpass()


# In[2]:


import os

os.environ["KAY_API_KEY"] = KAY_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# In[3]:


from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import KayAiRetriever
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")
retriever = KayAiRetriever.create(
    dataset_id="company", data_types=["PressRelease"], num_contexts=6
)
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)


# In[5]:


# More sample questions in the Playground on https://kay.ai
questions = [
    "How is the healthcare industry adopting generative AI tools?",
    # "What are some recent challenges faced by the renewable energy sector?",
]
chat_history = []

for question in questions:
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")
