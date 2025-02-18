#!/usr/bin/env python
# coding: utf-8

# # C Transformers
#
# The [C Transformers](https://github.com/marella/ctransformers) library provides Python bindings for GGML models.
#
# This example goes over how to use LangChain to interact with `C Transformers` [models](https://github.com/marella/ctransformers#supported-models).

# **Install**

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  ctransformers")


# **Load Model**

# In[ ]:


from langchain_community.llms import CTransformers

llm = CTransformers(model="marella/gpt-2-ggml")


# **Generate Text**

# In[ ]:


print(llm.invoke("AI is going to"))


# **Streaming**

# In[ ]:


from langchain_core.callbacks import StreamingStdOutCallbackHandler

llm = CTransformers(
    model="marella/gpt-2-ggml", callbacks=[StreamingStdOutCallbackHandler()]
)

response = llm.invoke("AI is going to")


# **LLMChain**

# In[ ]:


from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

template = """Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(prompt=prompt, llm=llm)

response = llm_chain.run("What is AI?")
