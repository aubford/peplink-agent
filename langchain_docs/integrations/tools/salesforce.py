#!/usr/bin/env python
# coding: utf-8

# # Salesforce
# 
# Tools for interacting with Salesforce.
# 
# ## Overview
# 
# This notebook provides examples of interacting with Salesforce using LangChain.
# 

# ## Setup
# 
# 1. Install the required dependencies:
# ```bash
#    pip install langchain-salesforce
# ```
# 
# 2. Set up your Salesforce credentials as environment variables:
# 
# ```bash
#    export SALESFORCE_USERNAME="your-username"
#    export SALESFORCE_PASSWORD="your-password" 
#    export SALESFORCE_SECURITY_TOKEN="your-security-token"
#    export SALESFORCE_DOMAIN="test" # Use 'test' for sandbox, remove for production
# ```
# 
# These environment variables will be automatically picked up by the integration.
# 
# ## Getting Your Security Token
# If you need a security token:
# 1. Log into Salesforce
# 2. Go to Settings
# 3. Click on "Reset My Security Token" under "My Personal Information"
# 4. Check your email for the new token

# ## Instantiation

# In[ ]:


import os

from langchain_salesforce import SalesforceTool

username = os.getenv("SALESFORCE_USERNAME", "your-username")
password = os.getenv("SALESFORCE_PASSWORD", "your-password")
security_token = os.getenv("SALESFORCE_SECURITY_TOKEN", "your-security-token")
domain = os.getenv("SALESFORCE_DOMAIN", "login")

tool = SalesforceTool(
    username=username, password=password, security_token=security_token, domain=domain
)


# ## Invocation

# In[ ]:


def execute_salesforce_operation(
    operation, object_name=None, query=None, record_data=None, record_id=None
):
    """Executes a given Salesforce operation."""
    request = {"operation": operation}
    if object_name:
        request["object_name"] = object_name
    if query:
        request["query"] = query
    if record_data:
        request["record_data"] = record_data
    if record_id:
        request["record_id"] = record_id
    result = tool.run(request)
    return result


# ## Query
# This example queries Salesforce for 5 contacts.

# In[ ]:


query_result = execute_salesforce_operation(
    "query", query="SELECT Id, Name, Email FROM Contact LIMIT 5"
)


# ## Describe an Object
# Fetches metadata for a specific Salesforce object.

# In[ ]:


describe_result = execute_salesforce_operation("describe", object_name="Account")


# ## List Available Objects
# Retrieves all objects available in the Salesforce instance.

# In[ ]:


list_objects_result = execute_salesforce_operation("list_objects")


# ## Create a New Contact
# Creates a new contact record in Salesforce.

# In[ ]:


create_result = execute_salesforce_operation(
    "create",
    object_name="Contact",
    record_data={"LastName": "Doe", "Email": "doe@example.com"},
)


# ## Update a Contact
# Updates an existing contact record.

# In[ ]:


update_result = execute_salesforce_operation(
    "update",
    object_name="Contact",
    record_id="003XXXXXXXXXXXXXXX",
    record_data={"Email": "updated@example.com"},
)


# ## Delete a Contact
# Deletes a contact record from Salesforce.

# In[ ]:


delete_result = execute_salesforce_operation(
    "delete", object_name="Contact", record_id="003XXXXXXXXXXXXXXX"
)


# ## Chaining

# In[ ]:


from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_salesforce import SalesforceTool

tool = SalesforceTool(
    username=username, password=password, security_token=security_token, domain=domain
)

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = PromptTemplate.from_template(
    "What is the name of the contact with the id {contact_id}?"
)

chain = prompt | tool.invoke | llm

result = chain.invoke({"contact_id": "003XXXXXXXXXXXXXXX"})


# ## API reference
# [langchain-salesforce README](https://github.com/colesmcintosh/langchain-salesforce/blob/main/README.md)
