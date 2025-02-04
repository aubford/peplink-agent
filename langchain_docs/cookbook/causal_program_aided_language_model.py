#!/usr/bin/env python
# coding: utf-8

# # Causal program-aided language (CPAL) chain
# 
# The CPAL chain builds on the recent PAL to stop LLM hallucination. The problem with the PAL approach is that it hallucinates on a math problem with a nested chain of dependence. The innovation here is that this new CPAL approach includes causal structure to fix hallucination.
# 
# The original [PR's description](https://github.com/langchain-ai/langchain/pull/6255) contains a full overview.
# 
# Using the CPAL chain, the LLM translated this
# 
#     "Tim buys the same number of pets as Cindy and Boris."
#     "Cindy buys the same number of pets as Bill plus Bob."
#     "Boris buys the same number of pets as Ben plus Beth."
#     "Bill buys the same number of pets as Obama."
#     "Bob buys the same number of pets as Obama."
#     "Ben buys the same number of pets as Obama."
#     "Beth buys the same number of pets as Obama."
#     "If Obama buys one pet, how many pets total does everyone buy?"
# 
# 
# into this
# 
# ![complex-graph.png](/img/cpal_diagram.png).
# 
# Outline of code examples demoed in this notebook.
# 
# 1. CPAL's value against hallucination: CPAL vs PAL  
#     1.1 Complex narrative  
#     1.2 Unanswerable math word problem  
# 2. CPAL's three types of causal diagrams ([The Book of Why](https://en.wikipedia.org/wiki/The_Book_of_Why)).   
#     2.1 Mediator   
#     2.2 Collider   
#     2.3 Confounder   

# In[1]:


from IPython.display import SVG
from langchain_experimental.cpal.base import CPALChain
from langchain_experimental.pal_chain import PALChain
from langchain_openai import OpenAI

llm = OpenAI(temperature=0, max_tokens=512)
cpal_chain = CPALChain.from_univariate_prompt(llm=llm, verbose=True)
pal_chain = PALChain.from_math_prompt(llm=llm, verbose=True)


# ## CPAL's value against hallucination: CPAL vs PAL
# 
# Like PAL, CPAL intends to reduce large language model (LLM) hallucination.
# 
# The CPAL chain is different from the PAL chain for a couple of reasons.
# 
# CPAL adds a causal structure (or DAG) to link entity actions (or math expressions).
# The CPAL math expressions are modeling a chain of cause and effect relations, which can be intervened upon, whereas for the PAL chain math expressions are projected math identities.
# 

# ### 1.1 Complex narrative
# 
# Takeaway: PAL hallucinates, CPAL does not hallucinate.

# In[2]:


question = (
    "Tim buys the same number of pets as Cindy and Boris."
    "Cindy buys the same number of pets as Bill plus Bob."
    "Boris buys the same number of pets as Ben plus Beth."
    "Bill buys the same number of pets as Obama."
    "Bob buys the same number of pets as Obama."
    "Ben buys the same number of pets as Obama."
    "Beth buys the same number of pets as Obama."
    "If Obama buys one pet, how many pets total does everyone buy?"
)


# In[3]:


pal_chain.run(question)


# In[4]:


cpal_chain.run(question)


# In[5]:


# wait 20 secs to see display
cpal_chain.draw(path="web.svg")
SVG("web.svg")


# ### Unanswerable math
# 
# Takeaway: PAL hallucinates, where CPAL, rather than hallucinate, answers with _"unanswerable, narrative question and plot are incoherent"_

# In[6]:


question = (
    "Jan has three times the number of pets as Marcia."
    "Marcia has two more pets than Cindy."
    "If Cindy has ten pets, how many pets does Barak have?"
)


# In[7]:


pal_chain.run(question)


# In[8]:


try:
    cpal_chain.run(question)
except Exception as e_msg:
    print(e_msg)


# ### Basic math
# 
# #### Causal mediator

# In[9]:


question = (
    "Jan has three times the number of pets as Marcia. "
    "Marcia has two more pets than Cindy. "
    "If Cindy has four pets, how many total pets do the three have?"
)


# ---
# PAL

# In[10]:


pal_chain.run(question)


# ---
# CPAL

# In[11]:


cpal_chain.run(question)


# In[12]:


# wait 20 secs to see display
cpal_chain.draw(path="web.svg")
SVG("web.svg")


# ### Causal collider

# In[13]:


question = (
    "Jan has the number of pets as Marcia plus the number of pets as Cindy. "
    "Marcia has no pets. "
    "If Cindy has four pets, how many total pets do the three have?"
)


# In[14]:


cpal_chain.run(question)


# In[15]:


# wait 20 secs to see display
cpal_chain.draw(path="web.svg")
SVG("web.svg")


# ### Causal confounder

# In[16]:


question = (
    "Jan has the number of pets as Marcia plus the number of pets as Cindy. "
    "Marcia has two more pets than Cindy. "
    "If Cindy has four pets, how many total pets do the three have?"
)


# In[17]:


cpal_chain.run(question)


# In[18]:


# wait 20 secs to see display
cpal_chain.draw(path="web.svg")
SVG("web.svg")


# In[19]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

