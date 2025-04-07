#!/usr/bin/env python
# coding: utf-8

# # vLLM
# 
# [vLLM](https://vllm.readthedocs.io/en/latest/index.html) is a fast and easy-to-use library for LLM inference and serving, offering:
# 
# * State-of-the-art serving throughput 
# * Efficient management of attention key and value memory with PagedAttention
# * Continuous batching of incoming requests
# * Optimized CUDA kernels
# 
# This notebooks goes over how to use a LLM with langchain and vLLM.
# 
# To use, you should have the `vllm` python package installed.

# In[1]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  vllm -q')


# In[1]:


from langchain_community.llms import VLLM

llm = VLLM(
    model="mosaicml/mpt-7b",
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=128,
    top_k=10,
    top_p=0.95,
    temperature=0.8,
)

print(llm.invoke("What is the capital of France ?"))


# ## Integrate the model in an LLMChain

# In[3]:


from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who was the US president in the year the first Pokemon game was released?"

print(llm_chain.invoke(question))


# ## Distributed Inference
# 
# vLLM supports distributed tensor-parallel inference and serving. 
# 
# To run multi-GPU inference with the LLM class, set the `tensor_parallel_size` argument to the number of GPUs you want to use. For example, to run inference on 4 GPUs

# In[ ]:


from langchain_community.llms import VLLM

llm = VLLM(
    model="mosaicml/mpt-30b",
    tensor_parallel_size=4,
    trust_remote_code=True,  # mandatory for hf models
)

llm.invoke("What is the future of AI?")


# ## Quantization
# 
# vLLM supports `awq` quantization. To enable it, pass `quantization` to `vllm_kwargs`.

# In[ ]:


llm_q = VLLM(
    model="TheBloke/Llama-2-7b-Chat-AWQ",
    trust_remote_code=True,
    max_new_tokens=512,
    vllm_kwargs={"quantization": "awq"},
)


# ## OpenAI-Compatible Server
# 
# vLLM can be deployed as a server that mimics the OpenAI API protocol. This allows vLLM to be used as a drop-in replacement for applications using OpenAI API.
# 
# This server can be queried in the same format as OpenAI API.
# 
# ### OpenAI-Compatible Completion

# In[3]:


from langchain_community.llms import VLLMOpenAI

llm = VLLMOpenAI(
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1",
    model_name="tiiuae/falcon-7b",
    model_kwargs={"stop": ["."]},
)
print(llm.invoke("Rome is"))


# ## LoRA adapter
# LoRA adapters can be used with any vLLM model that implements `SupportsLoRA`.

# In[ ]:


from langchain_community.llms import VLLM
from vllm.lora.request import LoRARequest

llm = VLLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    max_new_tokens=300,
    top_k=1,
    top_p=0.90,
    temperature=0.1,
    vllm_kwargs={
        "gpu_memory_utilization": 0.5,
        "enable_lora": True,
        "max_model_len": 350,
    },
)
LoRA_ADAPTER_PATH = "path/to/adapter"
lora_adapter = LoRARequest("lora_adapter", 1, LoRA_ADAPTER_PATH)

print(
    llm.invoke("What are some popular Korean street foods?", lora_request=lora_adapter)
)

