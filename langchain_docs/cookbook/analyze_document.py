#!/usr/bin/env python
# coding: utf-8

# # Analyze a single long document
#
# The AnalyzeDocumentChain takes in a single document, splits it up, and then runs it through a CombineDocumentsChain.

# In[3]:


with open("../docs/docs/modules/state_of_the_union.txt") as f:
    state_of_the_union = f.read()


# In[7]:


from langchain.chains import AnalyzeDocumentChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# In[8]:


from langchain.chains.question_answering import load_qa_chain

qa_chain = load_qa_chain(llm, chain_type="map_reduce")


# In[9]:


qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)


# In[10]:


qa_document_chain.run(
    input_document=state_of_the_union,
    question="what did the president say about justice breyer?",
)
