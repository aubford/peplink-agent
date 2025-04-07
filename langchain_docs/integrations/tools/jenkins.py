#!/usr/bin/env python
# coding: utf-8

# # Jenkins
# 
# Tools for interacting with [Jenkins](https://www.jenkins.io/).
# 

# ## Overview
# 
# The `langchain-jenkins` package allows you to execute and control CI/CD pipelines with
# Jenkins.

# ### Setup
# 
# Install `langchain-jenkins`:

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet langchain-jenkins')


# ### Credentials
# 
# You'll need to setup or obtain authorization to access Jenkins server.

# In[ ]:


import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("PASSWORD")


# ## Instantiation
# To disable the SSL Verify, set `os.environ["PYTHONHTTPSVERIFY"] = "0"`

# In[ ]:


from langchain_jenkins import JenkinsAPIWrapper, JenkinsJobRun

tools = [
    JenkinsJobRun(
        api_wrapper=JenkinsAPIWrapper(
            jenkins_server="https://example.com",
            username="admin",
            password=os.environ["PASSWORD"],
        )
    )
]


# ## Invocation
# You can now call invoke and pass arguments.

# 1. Create the Jenkins job

# In[ ]:


jenkins_job_content = ""
src_file = "job1.xml"
with open(src_file) as fread:
    jenkins_job_content = fread.read()
tools[0].invoke({"job": "job01", "config_xml": jenkins_job_content, "action": "create"})


# 2. Run the Jenkins Job

# In[ ]:


tools[0].invoke({"job": "job01", "parameters": {}, "action": "run"})


# 3. Get job info

# In[ ]:


resp = tools[0].invoke({"job": "job01", "number": 1, "action": "status"})
if not resp["inProgress"]:
    print(resp["result"])


# 4. Delete the jenkins job

# In[ ]:


tools[0].invoke({"job": "job01", "action": "delete"})


# ## Chaining
# 
# TODO.
# 

# ## API reference
# 
# For detailed documentation [API reference](https://python.langchain.com/docs/integrations/tools/jenkins/)
