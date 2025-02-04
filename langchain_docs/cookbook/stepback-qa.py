#!/usr/bin/env python
# coding: utf-8

# # Step-Back Prompting (Question-Answering)
# 
# One prompting technique called "Step-Back" prompting can improve performance on complex questions by first asking a "step back" question. This can be combined with regular question-answering applications by then doing retrieval on both the original and step-back question.
# 
# Read the paper [here](https://arxiv.org/abs/2310.06117)
# 
# See an excellent blog post on this by Cobus Greyling [here](https://cobusgreyling.medium.com/a-new-prompt-engineering-technique-has-been-introduced-called-step-back-prompting-b00e8954cacb)
# 
# In this cookbook we will replicate this technique. We modify the prompts used slightly to work better with chat models.

# In[85]:


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI


# In[86]:


# Few Shot Examples
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)


# In[87]:


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)


# In[88]:


question_gen = prompt | ChatOpenAI(temperature=0) | StrOutputParser()


# In[182]:


question = "was chatgpt around while trump was president?"


# In[183]:


question_gen.invoke({"question": question})


# In[190]:


from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

search = DuckDuckGoSearchAPIWrapper(max_results=4)


def retriever(query):
    return search.run(query)


# In[191]:


retriever(question)


# In[192]:


retriever(question_gen.invoke({"question": question}))


# In[193]:


# response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""
# response_prompt = ChatPromptTemplate.from_template(response_prompt_template)


# In[203]:


from langchain import hub

response_prompt = hub.pull("langchain-ai/stepback-answer")


# In[204]:


chain = (
    {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # Retrieve context using the step-back question
        "step_back_context": question_gen | retriever,
        # Pass on the question
        "question": lambda x: x["question"],
    }
    | response_prompt
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)


# In[205]:


chain.invoke({"question": question})


# ## Baseline

# In[206]:


response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

{normal_context}

Original Question: {question}
Answer:"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)


# In[207]:


chain = (
    {
        # Retrieve context using the normal question (only the first 3 results)
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # Pass on the question
        "question": lambda x: x["question"],
    }
    | response_prompt
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)


# In[208]:


chain.invoke({"question": question})


# In[ ]:




