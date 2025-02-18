#!/usr/bin/env python
# coding: utf-8

# In[2]:


from langchain_community.embeddings import AscendEmbeddings

model = AscendEmbeddings(
    model_path="/root/.cache/modelscope/hub/yangjhchs/acge_text_embedding",
    device_id=0,
    query_instruction="Represend this sentence for searching relevant passages: ",
)
emb = model.embed_query("hellow")
print(emb)


# In[3]:


doc_embs = model.embed_documents(
    ["This is a content of the document", "This is another document"]
)
print(doc_embs)


# In[4]:


model.aembed_query("hellow")


# In[5]:


await model.aembed_query("hellow")


# In[6]:


model.aembed_documents(
    ["This is a content of the document", "This is another document"]
)


# In[7]:


await model.aembed_documents(
    ["This is a content of the document", "This is another document"]
)


# In[ ]:
