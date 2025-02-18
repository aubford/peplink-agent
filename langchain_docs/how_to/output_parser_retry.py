#!/usr/bin/env python
# coding: utf-8

# # How to retry when a parsing error occurs
#
# While in some cases it is possible to fix any parsing mistakes by only looking at the output, in other cases it isn't. An example of this is when the output is not just in the incorrect format, but is partially complete. Consider the below example.

# In[1]:


from langchain.output_parsers import OutputFixingParser
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from pydantic import BaseModel, Field


# In[2]:


template = """Based on the user question, provide an Action and Action Input for what step should be taken.
{format_instructions}
Question: {query}
Response:"""


class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")


parser = PydanticOutputParser(pydantic_object=Action)


# In[3]:


prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


# In[4]:


prompt_value = prompt.format_prompt(query="who is leo di caprios gf?")


# In[5]:


bad_response = '{"action": "search"}'


# If we try to parse this response as is, we will get an error:

# In[6]:


try:
    parser.parse(bad_response)
except OutputParserException as e:
    print(e)


# If we try to use the `OutputFixingParser` to fix this error, it will be confused - namely, it doesn't know what to actually put for action input.

# In[7]:


fix_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())


# In[8]:


fix_parser.parse(bad_response)


# Instead, we can use the RetryOutputParser, which passes in the prompt (as well as the original output) to try again to get a better response.

# In[9]:


from langchain.output_parsers import RetryOutputParser


# In[10]:


retry_parser = RetryOutputParser.from_llm(parser=parser, llm=OpenAI(temperature=0))


# In[11]:


retry_parser.parse_with_prompt(bad_response, prompt_value)


# We can also add the RetryOutputParser easily with a custom chain which transform the raw LLM/ChatModel output into a more workable format.

# In[1]:


from langchain_core.runnables import RunnableLambda, RunnableParallel

completion_chain = prompt | OpenAI(temperature=0)

main_chain = RunnableParallel(
    completion=completion_chain, prompt_value=prompt
) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))


main_chain.invoke({"query": "who is leo di caprios gf?"})


# Find out api documentation for [RetryOutputParser](https://python.langchain.com/api_reference/langchain/output_parsers/langchain.output_parsers.retry.RetryOutputParser.html#langchain.output_parsers.retry.RetryOutputParser).

# In[ ]:
