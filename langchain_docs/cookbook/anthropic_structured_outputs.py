#!/usr/bin/env python
# coding: utf-8

# ## Tool Use with Anthropic API for structured outputs
#
# Anthropic API recently added tool use.
#
# This is very useful for structured output.

# In[ ]:


get_ipython().system(" pip install -U langchain-anthropic")


# In[ ]:


# Optional
import os

# os.environ['LANGSMITH_TRACING'] = 'true' # enables tracing
# os.environ['LANGSMITH_API_KEY'] = <your-api-key>


# `How can we use tools to produce structured output?`
#
# Function call / tool use just generates a payload.
#
# Payload often a JSON string, which can be pass to an API or, in this case, a parser to produce structured output.
#
# LangChain has `llm.with_structured_output(schema)` to make it very easy to produce structured output that matches `schema`.
#
# ![Screenshot 2024-04-03 at 10.16.57 PM.png](attachment:83c97bfe-b9b2-48ef-95cf-06faeebaa048.png)

# In[ ]:


from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


# Data model
class code(BaseModel):
    """Code output"""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


# LLM
llm = ChatAnthropic(
    model="claude-3-opus-20240229",
    default_headers={"anthropic-beta": "tools-2024-04-04"},
)

# Structured output, including raw will capture raw output and parser errors
structured_llm = llm.with_structured_output(code, include_raw=True)
code_output = structured_llm.invoke(
    "Write a python program that prints the string 'hello world' and tell me how it works in a sentence"
)


# In[2]:


# Initial reasoning stage
code_output["raw"].content[0]


# In[3]:


# Tool call
code_output["raw"].content[1]


# In[4]:


# JSON str
code_output["raw"].content[1]["input"]


# In[5]:


# Error
error = code_output["parsing_error"]
error


# In[6]:


# Result
parsed_result = code_output["parsed"]


# In[7]:


parsed_result.prefix


# In[8]:


parsed_result.imports


# In[9]:


parsed_result.code


# ## More challenging example
#
# Motivating example for tool use / structured outputs.
#
# ![code-gen.png](attachment:bb6c7126-7667-433f-ba50-56107b0341bd.png)

# Here are some docs that we want to answer code questions about.

# In[10]:


from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

# LCEL docs
url = "https://python.langchain.com/docs/expression_language/"
loader = RecursiveUrlLoader(
    url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

# Sort the list based on the URLs and get the text
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)


# Problem:
#
# `What if we want to enforce tool use?`
#
# We can use fallbacks.
#
# Let's select a code gen prompt that -- from some of my testing -- does not correctly invoke the tool.
#
# We can see if we can correct from this.

# In[12]:


# This code gen prompt invokes tool use
code_gen_prompt_working = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """<instructions> You are a coding assistant with expertise in LCEL, LangChain expression language. \n 
    Here is the LCEL documentation:  \n ------- \n  {context} \n ------- \n Answer the user  question based on the \n 
    above provided documentation. Ensure any code you provide can be executed with all required imports and variables \n
    defined. Structure your answer: 1) a prefix describing the code solution, 2) the imports, 3) the functioning code block. \n
    Invoke the code tool to structure the output correctly. </instructions> \n Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)

# This code gen prompt does not invoke tool use
code_gen_prompt_bad = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant with expertise in LCEL, LangChain expression language. \n 
    Here is a full set of LCEL documentation:  \n ------- \n  {context} \n ------- \n Answer the user 
    question based on the above provided documentation. Ensure any code you provide can be executed \n 
    with all required imports and variables defined. Structure your answer with a description of the code solution. \n
    Then list the imports. And finally list the functioning code block. Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)


# Data model
class code(BaseModel):
    """Code output"""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")
    description = "Schema for code solutions to questions about LCEL."


# LLM
llm = ChatAnthropic(
    model="claude-3-opus-20240229",
    default_headers={"anthropic-beta": "tools-2024-04-04"},
)

# Structured output
# Include raw will capture raw output and parser errors
structured_llm = llm.with_structured_output(code, include_raw=True)


# Check for errors
def check_claude_output(tool_output):
    """Check for parse error or failure to call the tool"""

    # Error with parsing
    if tool_output["parsing_error"]:
        # Report back output and parsing errors
        print("Parsing error!")
        raw_output = str(code_output["raw"].content)
        error = tool_output["parsing_error"]
        raise ValueError(
            f"Error parsing your output! Be sure to invoke the tool. Output: {raw_output}. \n Parse error: {error}"
        )

    # Tool was not invoked
    elif not tool_output["parsed"]:
        print("Failed to invoke tool!")
        raise ValueError(
            "You did not use the provided tool! Be sure to invoke the tool to structure the output."
        )
    return tool_output


# Chain with output check
code_chain = code_gen_prompt_bad | structured_llm | check_claude_output


# Let's add a check and re-try.

# In[13]:


def insert_errors(inputs):
    """Insert errors in the messages"""

    # Get errors
    error = inputs["error"]
    messages = inputs["messages"]
    messages += [
        (
            "user",
            f"Retry. You are required to fix the parsing errors: {error} \n\n You must invoke the provided tool.",
        )
    ]
    return {
        "messages": messages,
        "context": inputs["context"],
    }


# This will be run as a fallback chain
fallback_chain = insert_errors | code_chain
N = 3  # Max re-tries
code_chain_re_try = code_chain.with_fallbacks(
    fallbacks=[fallback_chain] * N, exception_key="error"
)


# In[14]:


# Test
messages = [("user", "How do I build a RAG chain in LCEL?")]
code_output_lcel = code_chain_re_try.invoke(
    {"context": concatenated_content, "messages": messages}
)


# In[15]:


parsed_result_lcel = code_output_lcel["parsed"]


# In[16]:


parsed_result_lcel.prefix


# In[17]:


parsed_result_lcel.imports


# In[18]:


parsed_result_lcel.code


# Example trace catching an error and correcting:
#
# https://smith.langchain.com/public/f06e62cb-2fac-46ae-80cd-0470b3155eae/r

# In[ ]:
