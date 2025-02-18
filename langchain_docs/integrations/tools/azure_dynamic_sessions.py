#!/usr/bin/env python
# coding: utf-8

# # Azure Container Apps dynamic sessions
#
# Azure Container Apps dynamic sessions provides a secure and scalable way to run a Python code interpreter in Hyper-V isolated sandboxes. This allows your agents to run potentially untrusted code in a secure environment. The code interpreter environment includes many popular Python packages, such as NumPy, pandas, and scikit-learn. See the [Azure Container App docs](https://learn.microsoft.com/en-us/azure/container-apps/sessions-code-interpreter) for more info on how sessions work.
#
# ## Setup
#
# By default, the `SessionsPythonREPLTool` tool uses `DefaultAzureCredential` to authenticate with Azure. Locally, it'll use your credentials from the Azure CLI or VS Code. Install the Azure CLI and log in with `az login` to authenticate.
#
# To use the code interpreter you'll also need to create a session pool, which you can do by following the instructions [here](https://learn.microsoft.com/en-us/azure/container-apps/sessions-code-interpreter?tabs=azure-cli#create-a-session-pool-with-azure-cli). Once that's done you should have a pool management session endpoint, which you'll need to set below:

# In[1]:


import getpass

POOL_MANAGEMENT_ENDPOINT = getpass.getpass()


# You'll also need to install the `langchain-azure-dynamic-sessions` package:

# In[ ]:


get_ipython().run_line_magic(
    "pip",
    "install -qU langchain-azure-dynamic-sessions langchain-openai langchainhub langchain langchain-community",
)


# ## Use tool
#
# Instantiate and use tool:

# In[2]:


from langchain_azure_dynamic_sessions import SessionsPythonREPLTool

tool = SessionsPythonREPLTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
tool.invoke("6 * 7")


# Invoking the tool will return a json string with the result of the code, along with any stdout and stderr outputs. To get the raw dictionary results, use the `execute()` method:

# In[3]:


tool.execute("6 * 7")


# ## Upload data
#
# If we want to perform computation over specific data, we can use the `upload_file()` functionality to upload data to our session. You can upload data either via the `data: BinaryIO` arg or via the `local_file_path: str` arg (which points to a local file on your system). The data is automatically uploaded to the "/mnt/data/" directory in the sessions container. You can get the full file path via the upload metadata returned by `upload_file()`.

# In[31]:


import io
import json

data = {"important_data": [1, 10, -1541]}
binary_io = io.BytesIO(json.dumps(data).encode("ascii"))

upload_metadata = tool.upload_file(
    data=binary_io, remote_file_path="important_data.json"
)

code = f"""
import json

with open("{upload_metadata.full_path}") as f:
    data = json.load(f)

sum(data['important_data'])
"""
tool.execute(code)


# ## Handling image results
#
# Dynamic sessions results can include image outputs as base64 encoded strings. In these cases the value of 'result' will be a dictionary with keys "type" (which will be "image"), "format (the format of the image), and "base64_data".

# In[9]:


code = """
import numpy as np
import matplotlib.pyplot as plt

# Generate values for x from -1 to 1
x = np.linspace(-1, 1, 400)

# Calculate the sine of each x value
y = np.sin(x)

# Create the plot
plt.plot(x, y)

# Add title and labels
plt.title('Plot of sin(x) from -1 to 1')
plt.xlabel('x')
plt.ylabel('sin(x)')

# Show the plot
plt.grid(True)
plt.show()
"""

result = tool.execute(code)
result["result"].keys()


# In[10]:


result["result"]["type"], result["result"]["format"]


# We can decode the image data and display it:

# In[14]:


import base64
import io

from IPython.display import display
from PIL import Image

base64_str = result["result"]["base64_data"]
img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
display(img)


# ## Simple agent example

# In[20]:


from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_azure_dynamic_sessions import SessionsPythonREPLTool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_tool_calling_agent(llm, [tool], prompt)

agent_executor = AgentExecutor(
    agent=agent, tools=[tool], verbose=True, handle_parsing_errors=True
)

response = agent_executor.invoke(
    {
        "input": "what's sin of pi . if it's negative generate a random number between 0 and 5. if it's positive between 5 and 10."
    }
)


# ## LangGraph data analyst agent
#
# For a more complex agent example check out the LangGraph data analyst example https://github.com/langchain-ai/langchain/blob/master/cookbook/azure_container_apps_dynamic_sessions_data_analyst.ipynb

# In[ ]:
