#!/usr/bin/env python
# coding: utf-8

# # DSPy
# 
# >[DSPy](https://github.com/stanfordnlp/dspy) is a fantastic framework for LLMs that introduces an automatic compiler that teaches LMs how to conduct the declarative steps in your program. Specifically, the DSPy compiler will internally trace your program and then craft high-quality prompts for large LMs (or train automatic finetunes for small LMs) to teach them the steps of your task.
# 
# Thanks to [Omar Khattab](https://twitter.com/lateinteraction) we have an integration! It works with any LCEL chains with some minor modifications.
# 
# This short tutorial demonstrates how this proof-of-concept feature works. *This will not give you the full power of DSPy or LangChain yet, but we will expand it if there's high demand.*
# 
# Note: this was slightly modified from the original example Omar wrote for DSPy. If you are interested in LangChain \<\> DSPy but coming from the DSPy side, I'd recommend checking that out. You can find that [here](https://github.com/stanfordnlp/dspy/blob/main/examples/tweets/compiling_langchain.ipynb).
# 
# Let's take a look at an example. In this example we will make a simple RAG pipeline. We will use DSPy to "compile" our program and learn an optimized prompt.
# 
# This example uses the `ColBERTv2` model.
# See the [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) paper.
# 
# 
# ## Install dependencies
# 
# !pip install -U dspy-ai 
# !pip install -U openai jinja2
# !pip install -U langchain langchain-community langchain-openai langchain-core

# ## Setup
# 
# We will be using OpenAI, so we should set an API key

# In[ ]:


import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()


# We can now set up our retriever. For our retriever we will use a ColBERT retriever through DSPy, though this will work with any retriever.

# In[1]:


import dspy

colbertv2 = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")


# In[2]:


from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_openai import OpenAI

set_llm_cache(SQLiteCache(database_path="cache.db"))

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)


def retrieve(inputs):
    return [doc["text"] for doc in colbertv2(inputs["question"], k=5)]


# In[22]:


colbertv2("cycling")


# ## Normal LCEL
# 
# First, let's create a simple RAG pipeline with LCEL like we would normally.
# 
# For illustration, let's tackle the following task.
# 
# **Task:** Build a RAG system for generating informative tweets.
# 
# - **Input:** A factual question, which may be fairly complex.
#  
# - **Output:** An engaging tweet that correctly answers the question from the retrieved info.
#  
# Let's use LangChain's expression language (LCEL) to illustrate this. Any prompt here will do, we will optimize the final prompt with DSPy.
# 
# Considering that, let's just keep it to the barebones: **Given \{context\}, answer the question \{question\} as a tweet.**

# In[3]:


# From LangChain, import standard modules for prompting.
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Just a simple prompt for this task. It's fine if it's complex too.
prompt = PromptTemplate.from_template(
    "Given {context}, answer the question `{question}` as a tweet."
)

# This is how you'd normally build a chain with LCEL. This chain does retrieval then generation (RAG).
vanilla_chain = (
    RunnablePassthrough.assign(context=retrieve) | prompt | llm | StrOutputParser()
)


# ## LCEL \<\> DSPy
# 
# In order to use LangChain with DSPy, you need to make two minor modifications
# 
# **LangChainPredict**
# 
# You need to change from doing `prompt | llm` to using `LangChainPredict(prompt, llm)` from `dspy`. 
# 
# This is a wrapper which will bind your prompt and llm together so you can optimize them
# 
# **LangChainModule**
# 
# This is a wrapper which wraps your final LCEL chain so that DSPy can optimize the whole thing

# In[4]:


# From DSPy, import the modules that know how to interact with LangChain LCEL.
from dspy.predict.langchain import LangChainModule, LangChainPredict

# This is how to wrap it so it behaves like a DSPy program.
# Just Replace every pattern like `prompt | llm` with `LangChainPredict(prompt, llm)`.
zeroshot_chain = (
    RunnablePassthrough.assign(context=retrieve)
    | LangChainPredict(prompt, llm)
    | StrOutputParser()
)
# Now we wrap it in LangChainModule
zeroshot_chain = LangChainModule(
    zeroshot_chain
)  # then wrap the chain in a DSPy module.


# ## Trying the Module
# 
# After this, we can use it as both a LangChain runnable and a DSPy module!

# In[5]:


question = "In what region was Eddy Mazzoleni born?"

zeroshot_chain.invoke({"question": question})


# Ah that sounds about right! (It's technically not perfect: we asked for the region not the city. We can do better below.)
# 
# Inspecting questions and answers manually is very important to get a sense of your system. However, a good system designer always looks to iteratively benchmark their work to quantify progress!
# 
# To do this, we need two things: the metric we want to maximize and a (tiny) dataset of examples for our system.
# 
# Are there pre-defined metrics for good tweets? Should I label 100,000 tweets by hand? Probably not. We can easily do something reasonable, though, until you start getting data in production!

# ## Load Data
# 
# In order to compile our chain, we need a dataset to work with. This dataset just needs to be raw inputs and outputs. For our purposes, we will use HotPotQA dataset
# 
# Note: Notice that our dataset doesn't actually include any tweets! It only has questions and answers. That's OK, our metric will take care of evaluating outputs in tweet form.

# In[6]:


import dspy
from dspy.datasets import HotPotQA

# Load the dataset.
dataset = HotPotQA(
    train_seed=1,
    train_size=200,
    eval_seed=2023,
    dev_size=200,
    test_size=0,
    keep_details=True,
)

# Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
trainset = [x.without("id", "type").with_inputs("question") for x in dataset.train]
devset = [x.without("id", "type").with_inputs("question") for x in dataset.dev]
valset, devset = devset[:50], devset[50:]


# ## Define a metric
# 
# We now need to define a metric. This will be used to determine which runs were successful and we can learn from. Here we will use DSPy's metrics, though you can write your own.

# In[7]:


# Define the signature for autoamtic assessments.
class Assess(dspy.Signature):
    """Assess the quality of a tweet along the specified dimension."""

    context = dspy.InputField(desc="ignore if N/A")
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")


gpt4T = dspy.OpenAI(model="gpt-4-1106-preview", max_tokens=1000, model_type="chat")
METRIC = None


def metric(gold, pred, trace=None):
    question, answer, tweet = gold.question, gold.answer, pred.output
    context = colbertv2(question, k=5)

    engaging = "Does the assessed text make for a self-contained, engaging tweet?"
    faithful = "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context."
    correct = (
        f"The text above is should answer `{question}`. The gold answer is `{answer}`."
    )
    correct = f"{correct} Does the assessed text above contain the gold answer?"

    with dspy.context(lm=gpt4T):
        faithful = dspy.Predict(Assess)(
            context=context, assessed_text=tweet, assessment_question=faithful
        )
        correct = dspy.Predict(Assess)(
            context="N/A", assessed_text=tweet, assessment_question=correct
        )
        engaging = dspy.Predict(Assess)(
            context="N/A", assessed_text=tweet, assessment_question=engaging
        )

    correct, engaging, faithful = [
        m.assessment_answer.split()[0].lower() == "yes"
        for m in [correct, engaging, faithful]
    ]
    score = (correct + engaging + faithful) if correct and (len(tweet) <= 280) else 0

    if METRIC is not None:
        if METRIC == "correct":
            return correct
        if METRIC == "engaging":
            return engaging
        if METRIC == "faithful":
            return faithful

    if trace is not None:
        return score >= 3
    return score / 3.0


# ## Evaluate Baseline
# 
# Okay, let's evaluate the unoptimized "zero-shot" version of our chain, converted from our LangChain LCEL object.

# In[8]:


from dspy.evaluate.evaluate import Evaluate


# In[9]:


evaluate = Evaluate(
    metric=metric, devset=devset, num_threads=8, display_progress=True, display_table=5
)
evaluate(zeroshot_chain)


# Okay, cool. Our zeroshot_chain gets about 42.00% on the 150 questions from the devset.
# 
# The table above shows some examples. For instance:
# 
# - Question: Who was a producer who produced albums for both rock bands Juke Karten and Thirty Seconds to Mars?
# 
# - Tweet: Brian Virtue, who has worked with bands like Jane's Addiction and Velvet Revolver, produced albums for both Juke Kartel and Thirty Seconds to Mars, showcasing... [truncated]
# 
# - Metric: 1.0 (A tweet that is correct, faithful, and engaging!*)
# 
# footnote: * At least according to our metric, which is just a DSPy program, so it too can be optimized if you'd like! Topic for another notebook, though.

# ## Optimize
# 
# Now, let's optimize performance

# In[10]:


from dspy.teleprompt import BootstrapFewShotWithRandomSearch


# In[11]:


# Set up the optimizer. We'll use very minimal hyperparameters for this example.
# Just do random search with ~3 attempts, and in each attempt, bootstrap <= 3 traces.
optimizer = BootstrapFewShotWithRandomSearch(
    metric=metric, max_bootstrapped_demos=3, num_candidate_programs=3
)

# Now use the optimizer to *compile* the chain. This could take 5-10 minutes, unless it's cached.
optimized_chain = optimizer.compile(zeroshot_chain, trainset=trainset, valset=valset)


# ## Evaluating the optimized chain
# 
# Well, how good is this? Let's do some proper evals!

# In[13]:


evaluate(optimized_chain)


# Alright! We've improved our chain from 42% to nearly 50%!

# ## Inspect the optimized chain
# 
# So what actually happened to improve this? We can take a look at this by looking at the optimized chain. We can do this in two ways
# 
# ### Look at the prompt used
# 
# We can look at what prompt was actually used. We can do this by looking at `dspy.settings`.

# In[14]:


prompt_used, output = dspy.settings.langchain_history[-1]


# In[15]:


print(prompt_used)


# ### Look at the demos
# 
# The way this was optimized was that we collected examples (or "demos") to put in the prompt. We can inspect the optmized_chain to get a sense for what those are.

# In[20]:


demos = [
    eg
    for eg in optimized_chain.modules[0].demos
    if hasattr(eg, "augmented") and eg.augmented
]


# In[21]:


demos


# In[ ]:




