#!/usr/bin/env python
# coding: utf-8

# # Human-in-the-loop Tool Validation
#
# This walkthrough demonstrates how to add human validation to any Tool. We'll do this using the `HumanApprovalCallbackhandler`.
#
# Let's suppose we need to make use of the `ShellTool`. Adding this tool to an automated flow poses obvious risks. Let's see how we could enforce manual human approval of inputs going into this tool.
#
# **Note**: We generally recommend against using the `ShellTool`. There's a lot of ways to misuse it, and it's not required for most use cases. We employ it here only for demonstration purposes.

# In[1]:


from langchain.callbacks import HumanApprovalCallbackHandler
from langchain.tools import ShellTool


# In[19]:


tool = ShellTool()


# In[20]:


print(tool.run("echo Hello World!"))


# ## Adding Human Approval
# Adding the default `HumanApprovalCallbackHandler` to the tool will make it so that a user has to manually approve every input to the tool before the command is actually executed.

# In[10]:


tool = ShellTool(callbacks=[HumanApprovalCallbackHandler()])


# In[15]:


print(tool.run("ls /usr"))


# In[17]:


print(tool.run("ls /private"))


# ## Configuring Human Approval
#
# Let's suppose we have an agent that takes in multiple tools, and we want it to only trigger human approval requests on certain tools and certain inputs. We can configure out callback handler to do just this.

# In[ ]:


from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import OpenAI


# In[33]:


def _should_check(serialized_obj: dict) -> bool:
    # Only require approval on ShellTool.
    return serialized_obj.get("name") == "terminal"


def _approve(_input: str) -> bool:
    if _input == "echo 'Hello World'":
        return True
    msg = (
        "Do you approve of the following input? "
        "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no."
    )
    msg += "\n\n" + _input + "\n"
    resp = input(msg)
    return resp.lower() in ("yes", "y")


callbacks = [HumanApprovalCallbackHandler(should_check=_should_check, approve=_approve)]


# In[34]:


llm = OpenAI(temperature=0)
tools = load_tools(["wikipedia", "llm-math", "terminal"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)


# In[38]:


agent.run(
    "It's 2023 now. How many years ago did Konrad Adenauer become Chancellor of Germany.",
    callbacks=callbacks,
)


# In[36]:


agent.run("print 'Hello World' in the terminal", callbacks=callbacks)


# In[39]:


agent.run("list all directories in /private", callbacks=callbacks)


# In[ ]:
