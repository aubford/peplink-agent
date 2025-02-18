#!/usr/bin/env python
# coding: utf-8

# # Aphrodite Engine
#
# [Aphrodite](https://github.com/PygmalionAI/aphrodite-engine) is the open-source large-scale inference engine designed to serve thousands of users on the [PygmalionAI](https://pygmalion.chat) website.
#
# * Attention mechanism by vLLM for fast throughput and low latencies
# * Support for for many SOTA sampling methods
# * Exllamav2 GPTQ kernels for better throughput at lower batch sizes
#
# This notebooks goes over how to use a LLM with langchain and Aphrodite.
#
# To use, you should have the `aphrodite-engine` python package installed.

# In[ ]:


##Installing the langchain packages needed to use the integration
get_ipython().run_line_magic("pip", "install -qU langchain-community")


# In[ ]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet  aphrodite-engine==0.4.2"
)
# %pip list | grep aphrodite


# In[2]:


from langchain_community.llms import Aphrodite

llm = Aphrodite(
    model="PygmalionAI/pygmalion-2-7b",
    trust_remote_code=True,  # mandatory for hf models
    max_tokens=128,
    temperature=1.2,
    min_p=0.05,
    mirostat_mode=0,  # change to 2 to use mirostat
    mirostat_tau=5.0,
    mirostat_eta=0.1,
)

print(
    llm.invoke(
        '<|system|>Enter RP mode. You are Ayumu "Osaka" Kasuga.<|user|>Hey Osaka. Tell me about yourself.<|model|>'
    )
)


# ## Integrate the model in an LLMChain

# In[3]:


from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who was the US president in the year the first Pokemon game was released?"

print(llm_chain.run(question))


# ## Distributed Inference
#
# Aphrodite supports distributed tensor-parallel inference and serving.
#
# To run multi-GPU inference with the LLM class, set the `tensor_parallel_size` argument to the number of GPUs you want to use. For example, to run inference on 4 GPUs

# In[1]:


from langchain_community.llms import Aphrodite

llm = Aphrodite(
    model="PygmalionAI/mythalion-13b",
    tensor_parallel_size=4,
    trust_remote_code=True,  # mandatory for hf models
)

llm("What is the future of AI?")
