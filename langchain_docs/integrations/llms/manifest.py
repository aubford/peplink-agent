#!/usr/bin/env python
# coding: utf-8

# # Manifest
# 
# This notebook goes over how to use Manifest and LangChain.

# For more detailed information on `manifest`, and how to use it with local huggingface models like in this example, see https://github.com/HazyResearch/manifest
# 
# Another example of [using Manifest with Langchain](https://github.com/HazyResearch/manifest/blob/main/examples/langchain_chatgpt.html).

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  manifest-ml')


# In[2]:


from langchain_community.llms.manifest import ManifestWrapper
from manifest import Manifest


# In[ ]:


manifest = Manifest(
    client_name="huggingface", client_connection="http://127.0.0.1:5000"
)
print(manifest.client_pool.get_current_client().get_model_params())


# In[5]:


llm = ManifestWrapper(
    client=manifest, llm_kwargs={"temperature": 0.001, "max_tokens": 256}
)


# In[6]:


# Map reduce example
from langchain.chains.mapreduce import MapReduceChain
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter

_prompt = """Write a concise summary of the following:


{text}


CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(_prompt)

text_splitter = CharacterTextSplitter()

mp_chain = MapReduceChain.from_params(llm, prompt, text_splitter)


# In[7]:


with open("../../how_to/state_of_the_union.txt") as f:
    state_of_the_union = f.read()
mp_chain.run(state_of_the_union)


# ## Compare HF Models

# In[8]:


from langchain.model_laboratory import ModelLaboratory

manifest1 = ManifestWrapper(
    client=Manifest(
        client_name="huggingface", client_connection="http://127.0.0.1:5000"
    ),
    llm_kwargs={"temperature": 0.01},
)
manifest2 = ManifestWrapper(
    client=Manifest(
        client_name="huggingface", client_connection="http://127.0.0.1:5001"
    ),
    llm_kwargs={"temperature": 0.01},
)
manifest3 = ManifestWrapper(
    client=Manifest(
        client_name="huggingface", client_connection="http://127.0.0.1:5002"
    ),
    llm_kwargs={"temperature": 0.01},
)
llms = [manifest1, manifest2, manifest3]
model_lab = ModelLaboratory(llms)


# In[9]:


model_lab.compare("What color is a flamingo?")

