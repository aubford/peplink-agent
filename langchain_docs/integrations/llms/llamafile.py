#!/usr/bin/env python
# coding: utf-8

# # Llamafile
#
# [Llamafile](https://github.com/Mozilla-Ocho/llamafile) lets you distribute and run LLMs with a single file.
#
# Llamafile does this by combining [llama.cpp](https://github.com/ggerganov/llama.cpp) with [Cosmopolitan Libc](https://github.com/jart/cosmopolitan) into one framework that collapses all the complexity of LLMs down to a single-file executable (called a "llamafile") that runs locally on most computers, with no installation.
#
# ## Setup
#
# 1. Download a llamafile for the model you'd like to use. You can find many models in llamafile format on [HuggingFace](https://huggingface.co/models?other=llamafile). In this guide, we will download a small one, `TinyLlama-1.1B-Chat-v1.0.Q5_K_M`. Note: if you don't have `wget`, you can just download the model via this [link](https://huggingface.co/jartine/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile?download=true).
#
# ```bash
# wget https://huggingface.co/jartine/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile
# ```
#
# 2. Make the llamafile executable. First, if you haven't done so already, open a terminal. **If you're using MacOS, Linux, or BSD,** you'll need to grant permission for your computer to execute this new file using `chmod` (see below). **If you're on Windows,** rename the file by adding ".exe" to the end (model file should be named `TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile.exe`).
#
#
# ```bash
# chmod +x TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile  # run if you're on MacOS, Linux, or BSD
# ```
#
# 3. Run the llamafile in "server mode":
#
# ```bash
# ./TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile --server --nobrowser
# ```
#
# Now you can make calls to the llamafile's REST API. By default, the llamafile server listens at http://localhost:8080. You can find full server documentation [here](https://github.com/Mozilla-Ocho/llamafile/blob/main/llama.cpp/server/README.md#api-endpoints). You can interact with the llamafile directly via the REST API, but here we'll show how to interact with it using LangChain.
#

# ## Usage

# In[4]:


from langchain_community.llms.llamafile import Llamafile

llm = Llamafile()

llm.invoke("Tell me a joke")


# To stream tokens, use the `.stream(...)` method:

# In[6]:


query = "Tell me a joke"

for chunks in llm.stream(query):
    print(chunks, end="")

print()


# To learn more about the LangChain Expressive Language and the available methods on an LLM, see the [LCEL Interface](/docs/concepts/runnables)
