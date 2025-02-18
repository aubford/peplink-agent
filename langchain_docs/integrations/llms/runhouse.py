#!/usr/bin/env python
# coding: utf-8

# # Runhouse
#
# [Runhouse](https://github.com/run-house/runhouse) allows remote compute and data across environments and users. See the [Runhouse docs](https://www.run.house/docs).
#
# This example goes over how to use LangChain and [Runhouse](https://github.com/run-house/runhouse) to interact with models hosted on your own GPU, or on-demand GPUs on AWS, GCP, AWS, or Lambda.
#
# **Note**: Code uses `SelfHosted` name instead of the `Runhouse`.

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  runhouse")


# In[1]:


import runhouse as rh
from langchain.chains import LLMChain
from langchain_community.llms import SelfHostedHuggingFaceLLM, SelfHostedPipeline
from langchain_core.prompts import PromptTemplate


# In[4]:


# For an on-demand A100 with GCP, Azure, or Lambda
gpu = rh.cluster(name="rh-a10x", instance_type="A100:1", use_spot=False)

# For an on-demand A10G with AWS (no single A100s on AWS)
# gpu = rh.cluster(name='rh-a10x', instance_type='g5.2xlarge', provider='aws')

# For an existing cluster
# gpu = rh.cluster(ips=['<ip of the cluster>'],
#                  ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
#                  name='rh-a10x')


# In[5]:


template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)


# In[ ]:


llm = SelfHostedHuggingFaceLLM(
    model_id="gpt2", hardware=gpu, model_reqs=["pip:./", "transformers", "torch"]
)


# In[6]:


llm_chain = LLMChain(prompt=prompt, llm=llm)


# In[31]:


question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.run(question)


# You can also load more custom models through the SelfHostedHuggingFaceLLM interface:

# In[ ]:


llm = SelfHostedHuggingFaceLLM(
    model_id="google/flan-t5-small",
    task="text2text-generation",
    hardware=gpu,
)


# In[39]:


llm("What is the capital of Germany?")


# Using a custom load function, we can load a custom pipeline directly on the remote hardware:

# In[34]:


def load_pipeline():
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
    )

    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
    )
    return pipe


def inference_fn(pipeline, prompt, stop=None):
    return pipeline(prompt)[0]["generated_text"][len(prompt) :]


# In[ ]:


llm = SelfHostedHuggingFaceLLM(
    model_load_fn=load_pipeline, hardware=gpu, inference_fn=inference_fn
)


# In[36]:


llm("Who is the current US president?")


# You can send your pipeline directly over the wire to your model, but this will only work for small models (&lt;2 Gb), and will be pretty slow:

# In[ ]:


pipeline = load_pipeline()
llm = SelfHostedPipeline.from_pipeline(
    pipeline=pipeline, hardware=gpu, model_reqs=["pip:./", "transformers", "torch"]
)


# Instead, we can also send it to the hardware's filesystem, which will be much faster.

# In[ ]:


import pickle

rh.blob(pickle.dumps(pipeline), path="models/pipeline.pkl").save().to(
    gpu, path="models"
)

llm = SelfHostedPipeline.from_pipeline(pipeline="models/pipeline.pkl", hardware=gpu)
