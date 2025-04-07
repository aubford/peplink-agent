#!/usr/bin/env python
# coding: utf-8

# ---
# sidebar_position: 3
# keywords: [summarize, summarization, stuff, create_stuff_documents_chain]
# ---

# # How to summarize text in a single LLM call
# 
# LLMs can summarize and otherwise distill desired information from text, including large volumes of text. In many cases, especially for models with larger context windows, this can be adequately achieved via a single LLM call.
# 
# LangChain implements a simple [pre-built chain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.combine_documents.stuff.create_stuff_documents_chain.html) that "stuffs" a prompt with the desired context for summarization and other purposes. In this guide we demonstrate how to use the chain.

# ## Load chat model
# 
# Let's first load a [chat model](/docs/concepts/chat_models/):
# 
# import ChatModelTabs from "@theme/ChatModelTabs";
# 
# <ChatModelTabs
#   customVarName="llm"
# />
# 

# In[1]:


# | output: false
# | echo: false

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ## Load documents

# Next, we need some documents to summarize. Below, we generate some toy documents for illustrative purposes. See the document loader [how-to guides](/docs/how_to/#document-loaders) and [integration pages](/docs/integrations/document_loaders/) for additional sources of data. The [summarization tutorial](/docs/tutorials/summarization) also includes an example summarizing a blog post.

# In[2]:


from langchain_core.documents import Document

documents = [
    Document(page_content="Apples are red", metadata={"title": "apple_book"}),
    Document(page_content="Blueberries are blue", metadata={"title": "blueberry_book"}),
    Document(page_content="Bananas are yelow", metadata={"title": "banana_book"}),
]


# ## Load chain
# 
# Below, we define a simple prompt and instantiate the chain with our chat model and documents:

# In[3]:


from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Summarize this content: {context}")
chain = create_stuff_documents_chain(llm, prompt)


# ## Invoke chain
# 
# Because the chain is a [Runnable](/docs/concepts/runnables), it implements the usual methods for invocation:

# In[4]:


result = chain.invoke({"context": documents})
result


# ### Streaming
# 
# Note that the chain also supports streaming of individual output tokens:

# In[6]:


for chunk in chain.stream({"context": documents}):
    print(chunk, end="|")


# ## Next steps
# 
# See the summarization [how-to guides](/docs/how_to/#summarization) for additional summarization strategies, including those designed for larger volumes of text.
# 
# See also [this tutorial](/docs/tutorials/summarization) for more detail on summarization.
