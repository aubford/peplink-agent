#!/usr/bin/env python
# coding: utf-8

# # Improve document indexing with HyDE
# This notebook goes over how to use Hypothetical Document Embeddings (HyDE), as described in [this paper](https://arxiv.org/abs/2212.10496).
#
# At a high level, HyDE is an embedding technique that takes queries, generates a hypothetical answer, and then embeds that generated document and uses that as the final example.
#
# In order to use HyDE, we therefore need to provide a base embedding model, as well as an LLMChain that can be used to generate those documents. By default, the HyDE class comes with some default prompts to use (see the paper for more details on them), but we can also create our own.

# In[1]:


from langchain.chains import HypotheticalDocumentEmbedder, LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings


# In[2]:


base_embeddings = OpenAIEmbeddings()
llm = OpenAI()


#

# In[3]:


# Load with `web_search` prompt
embeddings = HypotheticalDocumentEmbedder.from_llm(llm, base_embeddings, "web_search")


# In[4]:


# Now we can use it as any embedding class!
result = embeddings.embed_query("Where is the Taj Mahal?")


# ## Multiple generations
# We can also generate multiple documents and then combine the embeddings for those. By default, we combine those by taking the average. We can do this by changing the LLM we use to generate documents to return multiple things.

# In[5]:


multi_llm = OpenAI(n=4, best_of=4)


# In[6]:


embeddings = HypotheticalDocumentEmbedder.from_llm(
    multi_llm, base_embeddings, "web_search"
)


# In[7]:


result = embeddings.embed_query("Where is the Taj Mahal?")


# ## Using our own prompts
# Besides using preconfigured prompts, we can also easily construct our own prompts and use those in the LLMChain that is generating the documents. This can be useful if we know the domain our queries will be in, as we can condition the prompt to generate text more similar to that.
#
# In the example below, let's condition it to generate text about a state of the union address (because we will use that in the next example).

# In[8]:


prompt_template = """Please answer the user's question about the most recent state of the union address
Question: {question}
Answer:"""
prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
llm_chain = LLMChain(llm=llm, prompt=prompt)


# In[9]:


embeddings = HypotheticalDocumentEmbedder(
    llm_chain=llm_chain, base_embeddings=base_embeddings
)


# In[10]:


result = embeddings.embed_query(
    "What did the president say about Ketanji Brown Jackson"
)


# ## Using HyDE
# Now that we have HyDE, we can use it as we would any other embedding class! Here is using it to find similar passages in the state of the union example.

# In[11]:


from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter

with open("../../state_of_the_union.txt") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)


# In[12]:


docsearch = Chroma.from_texts(texts, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)


# In[13]:


print(docs[0].page_content)


# In[ ]:
