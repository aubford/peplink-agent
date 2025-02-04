#!/usr/bin/env python
# coding: utf-8

# # CTranslate2

# **CTranslate2** is a C++ and Python library for efficient inference with Transformer models.
# 
# The project implements a custom runtime that applies many performance optimization techniques such as weights quantization, layers fusion, batch reordering, etc., to accelerate and reduce the memory usage of Transformer models on CPU and GPU.
# 
# Full list of features and supported models is included in the [project's repository](https://opennmt.net/CTranslate2/guides/transformers.html). To start, please check out the official [quickstart guide](https://opennmt.net/CTranslate2/quickstart.html).
# 
# To use, you should have `ctranslate2` python package installed.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  ctranslate2')


# To use a Hugging Face model with CTranslate2, it has to be first converted to CTranslate2 format using the `ct2-transformers-converter` command. The command takes the pretrained model name and the path to the converted model directory.

# In[2]:


# conversation can take several minutes
get_ipython().system('ct2-transformers-converter --model meta-llama/Llama-2-7b-hf --quantization bfloat16 --output_dir ./llama-2-7b-ct2 --force')


# In[3]:


from langchain_community.llms import CTranslate2

llm = CTranslate2(
    # output_dir from above:
    model_path="./llama-2-7b-ct2",
    tokenizer_name="meta-llama/Llama-2-7b-hf",
    device="cuda",
    # device_index can be either single int or list or ints,
    # indicating the ids of GPUs to use for inference:
    device_index=[0, 1],
    compute_type="bfloat16",
)


# ## Single call

# In[31]:


print(
    llm.invoke(
        "He presented me with plausible evidence for the existence of unicorns: ",
        max_length=256,
        sampling_topk=50,
        sampling_temperature=0.2,
        repetition_penalty=2,
        cache_static_prompt=False,
    )
)


# ## Multiple calls:

# In[34]:


print(
    llm.generate(
        ["The list of top romantic songs:\n1.", "The list of top rap songs:\n1."],
        max_length=128,
    )
)


# ## Integrate the model in an LLMChain

# In[46]:


from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

template = """{question}

Let's think step by step. """
prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who was the US president in the year the first Pokemon game was released?"

print(llm_chain.run(question))

