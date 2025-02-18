#!/usr/bin/env python
# coding: utf-8

# # E2B Data Analysis
#
# [E2B's cloud environments](https://e2b.dev) are great runtime sandboxes for LLMs.
#
# E2B's Data Analysis sandbox allows for safe code execution in a sandboxed environment. This is ideal for building tools such as code interpreters, or Advanced Data Analysis like in ChatGPT.
#
# E2B Data Analysis sandbox allows you to:
# - Run Python code
# - Generate charts via matplotlib
# - Install Python packages dynamically during runtime
# - Install system packages dynamically during runtime
# - Run shell commands
# - Upload and download files
#
# We'll create a simple OpenAI agent that will use E2B's Data Analysis sandbox to perform analysis on a uploaded files using Python.

# Get your OpenAI API key and [E2B API key here](https://e2b.dev/docs/getting-started/api-key) and set them as environment variables.
#
# You can find the full API documentation [here](https://e2b.dev/docs).
#

# You'll need to install `e2b` to get started:

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet  langchain e2b langchain-community"
)


# In[ ]:


from langchain_community.tools import E2BDataAnalysisTool


# In[1]:


import os

from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI

os.environ["E2B_API_KEY"] = "<E2B_API_KEY>"
os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"


# When creating an instance of the `E2BDataAnalysisTool`, you can pass callbacks to listen to the output of the sandbox. This is useful, for example, when creating more responsive UI. Especially with the combination of streaming output from LLMs.

# In[2]:


# Artifacts are charts created by matplotlib when `plt.show()` is called
def save_artifact(artifact):
    print("New matplotlib chart generated:", artifact.name)
    # Download the artifact as `bytes` and leave it up to the user to display them (on frontend, for example)
    file = artifact.download()
    basename = os.path.basename(artifact.name)

    # Save the chart to the `charts` directory
    with open(f"./charts/{basename}", "wb") as f:
        f.write(file)


e2b_data_analysis_tool = E2BDataAnalysisTool(
    # Pass environment variables to the sandbox
    env_vars={"MY_SECRET": "secret_value"},
    on_stdout=lambda stdout: print("stdout:", stdout),
    on_stderr=lambda stderr: print("stderr:", stderr),
    on_artifact=save_artifact,
)


# Upload an example CSV data file to the sandbox so we can analyze it with our agent. You can use for example [this file](https://storage.googleapis.com/e2b-examples/netflix.csv) about Netflix tv shows.

# In[3]:


with open("./netflix.csv") as f:
    remote_path = e2b_data_analysis_tool.upload_file(
        file=f,
        description="Data about Netflix tv shows including their title, category, director, release date, casting, age rating, etc.",
    )
    print(remote_path)


# Create a `Tool` object and initialize the Langchain agent.

# In[4]:


tools = [e2b_data_analysis_tool.as_tool()]

llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    handle_parsing_errors=True,
)


# Now we can ask the agent questions about the CSV file we uploaded earlier.

# In[5]:


agent.run(
    "What are the 5 longest movies on netflix released between 2000 and 2010? Create a chart with their lengths."
)


# E2B also allows you to install both Python and system (via `apt`) packages dynamically during runtime like this:

# In[6]:


# Install Python package
e2b_data_analysis_tool.install_python_packages("pandas")


# Additionally, you can download any file from the sandbox like this:

# In[7]:


# The path is a remote path in the sandbox
files_in_bytes = e2b_data_analysis_tool.download_file("/home/user/netflix.csv")


# Lastly, you can run any shell command inside the sandbox via `run_command`.

# In[8]:


# Install SQLite
e2b_data_analysis_tool.run_command("sudo apt update")
e2b_data_analysis_tool.install_system_packages("sqlite3")

# Check the SQLite version
output = e2b_data_analysis_tool.run_command("sqlite3 --version")
print("version: ", output["stdout"])
print("error: ", output["stderr"])
print("exit code: ", output["exit_code"])


# When your agent is finished, don't forget to close the sandbox

# In[9]:


e2b_data_analysis_tool.close()
