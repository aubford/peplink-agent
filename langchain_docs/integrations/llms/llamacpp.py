#!/usr/bin/env python
# coding: utf-8

# # Llama.cpp
#
# [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) is a Python binding for [llama.cpp](https://github.com/ggerganov/llama.cpp).
#
# It supports inference for [many LLMs](https://github.com/ggerganov/llama.cpp#description) models, which can be accessed on [Hugging Face](https://huggingface.co/TheBloke).
#
# This notebook goes over how to run `llama-cpp-python` within LangChain.
#
# **Note: new versions of `llama-cpp-python` use GGUF model files (see [here](https://github.com/abetlen/llama-cpp-python/pull/633)).**
#
# This is a breaking change.
#
# To convert existing GGML models to GGUF you can run the following in [llama.cpp](https://github.com/ggerganov/llama.cpp):
#
# ```
# python ./convert-llama-ggmlv3-to-gguf.py --eps 1e-5 --input models/openorca-platypus2-13b.ggmlv3.q4_0.bin --output models/openorca-platypus2-13b.gguf.q4_0.bin
# ```

# ## Installation
#
# There are different options on how to install the llama-cpp package:
# - CPU usage
# - CPU + GPU (using one of many BLAS backends)
# - Metal GPU (MacOS with Apple Silicon Chip)
#
# ### CPU only installation

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  llama-cpp-python")


# ### Installation with OpenBLAS / cuBLAS / CLBlast
#
# `llama.cpp` supports multiple BLAS backends for faster processing. Use the `FORCE_CMAKE=1` environment variable to force the use of cmake and install the pip package for the desired BLAS backend ([source](https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast)).
#
# Example installation with cuBLAS backend:

# In[ ]:


get_ipython().system(
    'CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python'
)


# **IMPORTANT**: If you have already installed the CPU only version of the package, you need to reinstall it from scratch. Consider the following command:

# In[ ]:


get_ipython().system(
    'CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir'
)


# ### Installation with Metal
#
# `llama.cpp` supports Apple silicon first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks. Use the `FORCE_CMAKE=1` environment variable to force the use of cmake and install the pip package for the Metal support ([source](https://github.com/abetlen/llama-cpp-python/blob/main/docs/install/macos.md)).
#
# Example installation with Metal Support:

# In[ ]:


get_ipython().system(
    'CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install llama-cpp-python'
)


# **IMPORTANT**: If you have already installed a cpu only version of the package, you need to reinstall it from scratch: consider the following command:

# In[ ]:


get_ipython().system(
    'CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir'
)


# ### Installation with Windows
#
# It is stable to install the `llama-cpp-python` library by compiling from the source. You can follow most of the instructions in the repository itself but there are some windows specific instructions which might be useful.
#
# Requirements to install the `llama-cpp-python`,
#
# - git
# - python
# - cmake
# - Visual Studio Community (make sure you install this with the following settings)
#     - Desktop development with C++
#     - Python development
#     - Linux embedded development with C++
#
# 1. Clone git repository recursively to get `llama.cpp` submodule as well
#
# ```
# git clone --recursive -j8 https://github.com/abetlen/llama-cpp-python.git
# ```
#
# 2. Open up a command Prompt and set the following environment variables.
#
#
# ```
# set FORCE_CMAKE=1
# set CMAKE_ARGS=-DLLAMA_CUBLAS=OFF
# ```
# If you have an NVIDIA GPU make sure `DLLAMA_CUBLAS` is set to `ON`
#
# #### Compiling and installing
#
# Now you can `cd` into the `llama-cpp-python` directory and install the package
#
# ```
# python -m pip install -e .
# ```

# **IMPORTANT**: If you have already installed a cpu only version of the package, you need to reinstall it from scratch: consider the following command:

# In[ ]:


get_ipython().system("python -m pip install -e . --force-reinstall --no-cache-dir")


# ## Usage

# Make sure you are following all instructions to [install all necessary model files](https://github.com/ggerganov/llama.cpp).
#
# You don't need an `API_TOKEN` as you will run the LLM locally.
#
# It is worth understanding which models are suitable to be used on the desired machine.
#
# [TheBloke's](https://huggingface.co/TheBloke) Hugging Face models have a `Provided files` section that exposes the RAM required to run models of different quantisation sizes and methods (eg: [Llama2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF#provided-files)).
#
# This [github issue](https://github.com/facebookresearch/llama/issues/425) is also relevant to find the right model for your machine.

# In[1]:


from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate


# **Consider using a template that suits your model! Check the models page on Hugging Face etc. to get a correct prompting template.**

# In[2]:


template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate.from_template(template)


# In[3]:


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


# ### CPU

# Example using a LLaMA 2 7B model

# In[ ]:


# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)


# In[13]:


question = """
Question: A rap battle between Stephen Colbert and John Oliver
"""
llm.invoke(question)


# Example using a LLaMA v1 model

# In[18]:


# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="./ggml-model-q4_0.bin", callback_manager=callback_manager, verbose=True
)


# In[16]:


llm_chain = prompt | llm


# In[17]:


question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
llm_chain.invoke({"question": question})


# ### GPU
#
# If the installation with BLAS backend was correct, you will see a `BLAS = 1` indicator in model properties.
#
# Two of the most important parameters for use with GPU are:
#
# - `n_gpu_layers` - determines how many layers of the model are offloaded to your GPU.
# - `n_batch` - how many tokens are processed in parallel.
#
# Setting these parameters correctly will dramatically improve the evaluation speed (see [wrapper code](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/llamacpp.py) for more details).

# In[ ]:


n_gpu_layers = (
    -1
)  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)


# In[5]:


llm_chain = prompt | llm
question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
llm_chain.invoke({"question": question})


# ### Metal
#
# If the installation with Metal was correct, you will see a `NEON = 1` indicator in model properties.
#
# Two of the most important GPU parameters are:
#
# - `n_gpu_layers` - determines how many layers of the model are offloaded to your Metal GPU.
# - `n_batch` - how many tokens are processed in parallel, default is 8, set to bigger number.
# - `f16_kv` - for some reason, Metal only support `True`, otherwise you will get error such as `Asserting on type 0
# GGML_ASSERT: .../ggml-metal.m:706: false && "not implemented"`
#
# Setting these parameters correctly will dramatically improve the evaluation speed (see [wrapper code](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/llamacpp.py) for more details).

# In[ ]:


n_gpu_layers = 1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)


# The console log will show the following log to indicate Metal was enable properly.
#
# ```
# ggml_metal_init: allocating
# ggml_metal_init: using MPS
# ...
# ```
#
# You also could check `Activity Monitor` by watching the GPU usage of the process, the CPU usage will drop dramatically after turn on `n_gpu_layers=1`.
#
# For the first call to the LLM, the performance may be slow due to the model compilation in Metal GPU.

# ### Grammars
#
# We can use [grammars](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) to constrain model outputs and sample tokens based on the rules defined in them.
#
# To demonstrate this concept, we've included [sample grammar files](https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/llms/grammars), that will be used in the examples below.
#
# Creating gbnf grammar files can be time-consuming, but if you have a use-case where output schemas are important, there are two tools that can help:
# - [Online grammar generator app](https://grammar.intrinsiclabs.ai/) that converts TypeScript interface definitions to gbnf file.
# - [Python script](https://github.com/ggerganov/llama.cpp/blob/master/examples/json-schema-to-grammar.py) for converting json schema to gbnf file. You can for example create `pydantic` object, generate its JSON schema using `.schema_json()` method, and then use this script to convert it to gbnf file.

# In the first example, supply the path to the specified `json.gbnf` file in order to produce JSON:

# In[ ]:


n_gpu_layers = 1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
    grammar_path="/Users/rlm/Desktop/Code/langchain-main/langchain/libs/langchain/langchain/llms/grammars/json.gbnf",
)


# In[7]:


get_ipython().run_cell_magic(
    "capture",
    "captured --no-stdout",
    'result = llm.invoke("Describe a person in JSON format:")\n',
)


# We can also supply `list.gbnf` to return a list:

# In[ ]:


n_gpu_layers = 1
n_batch = 512
llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
    grammar_path="/Users/rlm/Desktop/Code/langchain-main/langchain/libs/langchain/langchain/llms/grammars/list.gbnf",
)


# In[9]:


get_ipython().run_cell_magic(
    "capture",
    "captured --no-stdout",
    'result = llm.invoke("List of top-3 my favourite books:")\n',
)
