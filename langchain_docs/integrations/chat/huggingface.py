#!/usr/bin/env python
# coding: utf-8

# # ChatHuggingFace
#
# This will help you getting started with `langchain_huggingface` [chat models](/docs/concepts/chat_models). For detailed documentation of all `ChatHuggingFace` features and configurations head to the [API reference](https://python.langchain.com/api_reference/huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html). For a list of models supported by Hugging Face check out [this page](https://huggingface.co/models).
#
# ## Overview
# ### Integration details
#
# ### Integration details
#
# | Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [ChatHuggingFace](https://python.langchain.com/api_reference/huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html) | [langchain-huggingface](https://python.langchain.com/api_reference/huggingface/index.html) | ✅ | beta | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_huggingface?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_huggingface?style=flat-square&label=%20) |
#
# ### Model features
# | [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
#
# ## Setup
#
# To access Hugging Face models you'll need to create a Hugging Face account, get an API key, and install the `langchain-huggingface` integration package.
#
# ### Credentials
#
# Generate a [Hugging Face Access Token](https://huggingface.co/docs/hub/security-tokens) and store it as an environment variable: `HUGGINGFACEHUB_API_TOKEN`.

# In[ ]:


import getpass
import os

if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your token: ")


# ### Installation
#
# | Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [ChatHuggingFace](https://python.langchain.com/api_reference/huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html) | [langchain_huggingface](https://python.langchain.com/api_reference/huggingface/index.html) | ✅ | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_huggingface?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_huggingface?style=flat-square&label=%20) |
#
# ### Model features
# | [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
#
# ## Setup
#
# To access `langchain_huggingface` models you'll need to create a/an `Hugging Face` account, get an API key, and install the `langchain_huggingface` integration package.
#
# ### Credentials
#
# You'll need to have a [Hugging Face Access Token](https://huggingface.co/docs/hub/security-tokens) saved as an environment variable: `HUGGINGFACEHUB_API_TOKEN`.

# In[1]:


import getpass
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass(
    "Enter your Hugging Face API key: "
)


# In[5]:


get_ipython().run_line_magic(
    "pip",
    "install --upgrade --quiet  langchain-huggingface text-generation transformers google-search-results numexpr langchainhub sentencepiece jinja2 bitsandbytes accelerate",
)


# ## Instantiation
#
# You can instantiate a `ChatHuggingFace` model in two different ways, either from a `HuggingFaceEndpoint` or from a `HuggingFacePipeline`.

# ### `HuggingFaceEndpoint`

# In[10]:


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

chat_model = ChatHuggingFace(llm=llm)


# ### `HuggingFacePipeline`

# In[9]:


from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)

chat_model = ChatHuggingFace(llm=llm)


# ### Instatiating with Quantization
#
# To run a quantized version of your model, you can specify a `bitsandbytes` quantization config as follows:

# In[6]:


from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)


# and pass it to the `HuggingFacePipeline` as a part of its `model_kwargs`:

# In[ ]:


llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        return_full_text=False,
    ),
    model_kwargs={"quantization_config": quantization_config},
)

chat_model = ChatHuggingFace(llm=llm)


# ## Invocation

# In[11]:


from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(
        content="What happens when an unstoppable force meets an immovable object?"
    ),
]

ai_msg = chat_model.invoke(messages)


# In[12]:


print(ai_msg.content)


# ## API reference
#
# For detailed documentation of all `ChatHuggingFace` features and configurations head to the API reference: https://python.langchain.com/api_reference/huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html

# ## API reference
#
# For detailed documentation of all ChatHuggingFace features and configurations head to the API reference: https://python.langchain.com/api_reference/huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html
