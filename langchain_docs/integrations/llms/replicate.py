#!/usr/bin/env python
# coding: utf-8

# # Replicate
#
# >[Replicate](https://replicate.com/blog/machine-learning-needs-better-tools) runs machine learning models in the cloud. We have a library of open-source models that you can run with a few lines of code. If you're building your own machine learning models, Replicate makes it easy to deploy them at scale.
#
# This example goes over how to use LangChain to interact with `Replicate` [models](https://replicate.com/explore)

# ## Setup

# In[1]:


# magics to auto-reload external modules in case you are making changes to langchain while working on this notebook
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")


# To run this notebook, you'll need to create a [replicate](https://replicate.com) account and install the [replicate python client](https://github.com/replicate/replicate-python).

# In[4]:


get_ipython().system("poetry run pip install replicate")


# In[5]:


# get a token: https://replicate.com/account

from getpass import getpass

REPLICATE_API_TOKEN = getpass()


# In[6]:


import os

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN


# In[56]:


from langchain.chains import LLMChain
from langchain_community.llms import Replicate
from langchain_core.prompts import PromptTemplate


# ## Calling a model
#
# Find a model on the [replicate explore page](https://replicate.com/explore), and then paste in the model name and version in this format: model_name/version.
#
# For example, here is [`Meta Llama 3`](https://replicate.com/meta/meta-llama-3-8b-instruct).

# In[58]:


llm = Replicate(
    model="meta/meta-llama-3-8b-instruct",
    model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
)
prompt = """
User: Answer the following yes/no question by reasoning step by step. Can a dog drive a car?
Assistant:
"""
llm(prompt)


# As another example, for this [dolly model](https://replicate.com/replicate/dolly-v2-12b), click on the API tab. The model name/version would be: `replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5`
#
# Only the `model` param is required, but we can add other model params when initializing.
#
# For example, if we were running stable diffusion and wanted to change the image dimensions:
#
# ```
# Replicate(model="stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf", input={'image_dimensions': '512x512'})
# ```
#
# *Note that only the first output of a model will be returned.*

# In[20]:


llm = Replicate(
    model="replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5"
)


# In[21]:


prompt = """
Answer the following yes/no question by reasoning step by step.
Can a dog drive a car?
"""
llm(prompt)


# We can call any replicate model using this syntax. For example, we can call stable diffusion.

# In[22]:


text2image = Replicate(
    model="stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
    model_kwargs={"image_dimensions": "512x512"},
)


# In[23]:


image_output = text2image("A cat riding a motorcycle by Picasso")
image_output


# The model spits out a URL. Let's render it.

# In[24]:


get_ipython().system("poetry run pip install Pillow")


# In[ ]:


from io import BytesIO

import requests
from PIL import Image

response = requests.get(image_output)
img = Image.open(BytesIO(response.content))

img


# ## Streaming Response
# You can optionally stream the response as it is produced, which is helpful to show interactivity to users for time-consuming generations. See detailed docs on [Streaming](/docs/how_to/streaming_llm) for more information.

# In[26]:


from langchain_core.callbacks import StreamingStdOutCallbackHandler

llm = Replicate(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
)
prompt = """
User: Answer the following yes/no question by reasoning step by step. Can a dog drive a car?
Assistant:
"""
_ = llm.invoke(prompt)


# # Stop Sequences
# You can also specify stop sequences. If you have a definite stop sequence for the generation that you are going to parse with anyway, it is better (cheaper and faster!) to just cancel the generation once one or more stop sequences are reached, rather than letting the model ramble on till the specified `max_length`. Stop sequences work regardless of whether you are in streaming mode or not, and Replicate only charges you for the generation up until the stop sequence.

# In[27]:


import time

llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    model_kwargs={"temperature": 0.01, "max_length": 500, "top_p": 1},
)

prompt = """
User: What is the best way to learn python?
Assistant:
"""
start_time = time.perf_counter()
raw_output = llm.invoke(prompt)  # raw output, no stop
end_time = time.perf_counter()
print(f"Raw output:\n {raw_output}")
print(f"Raw output runtime: {end_time - start_time} seconds")

start_time = time.perf_counter()
stopped_output = llm.invoke(prompt, stop=["\n\n"])  # stop on double newlines
end_time = time.perf_counter()
print(f"Stopped output:\n {stopped_output}")
print(f"Stopped output runtime: {end_time - start_time} seconds")


# ## Chaining Calls
# The whole point of langchain is to... chain! Here's an example of how do that.

# In[28]:


from langchain.chains import SimpleSequentialChain


# First, let's define the LLM for this model as a flan-5, and text2image as a stable diffusion model.

# In[29]:


dolly_llm = Replicate(
    model="replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5"
)
text2image = Replicate(
    model="stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf"
)


# First prompt in the chain

# In[30]:


prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=dolly_llm, prompt=prompt)


# Second prompt to get the logo for company description

# In[31]:


second_prompt = PromptTemplate(
    input_variables=["company_name"],
    template="Write a description of a logo for this company: {company_name}",
)
chain_two = LLMChain(llm=dolly_llm, prompt=second_prompt)


# Third prompt, let's create the image based on the description output from prompt 2

# In[32]:


third_prompt = PromptTemplate(
    input_variables=["company_logo_description"],
    template="{company_logo_description}",
)
chain_three = LLMChain(llm=text2image, prompt=third_prompt)


# Now let's run it!

# In[33]:


# Run the chain specifying only the input variable for the first chain.
overall_chain = SimpleSequentialChain(
    chains=[chain, chain_two, chain_three], verbose=True
)
catchphrase = overall_chain.run("colorful socks")
print(catchphrase)


# In[ ]:


response = requests.get(
    "https://replicate.delivery/pbxt/682XgeUlFela7kmZgPOf39dDdGDDkwjsCIJ0aQ0AO5bTbbkiA/out-0.png"
)
img = Image.open(BytesIO(response.content))
img
