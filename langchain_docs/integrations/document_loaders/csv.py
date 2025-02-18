#!/usr/bin/env python
# coding: utf-8

# # CSV
#
# >A [comma-separated values (CSV)](https://en.wikipedia.org/wiki/Comma-separated_values) file is a delimited text file that uses a comma to separate values. Each line of the file is a data record. Each record consists of one or more fields, separated by commas.
#
# Load [csv](https://en.wikipedia.org/wiki/Comma-separated_values) data with a single row per document.

# In[1]:


from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path="./example_data/mlb_teams_2012.csv")

data = loader.load()

print(data)


# ## Customizing the csv parsing and loading
#
# See the [csv module](https://docs.python.org/3/library/csv.html) documentation for more information of what csv args are supported.

# In[4]:


loader = CSVLoader(
    file_path="./example_data/mlb_teams_2012.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["MLB Team", "Payroll in millions", "Wins"],
    },
)

data = loader.load()

print(data)


# ## Specify a column to identify the document source
#
# Use the `source_column` argument to specify a source for the document created from each row. Otherwise `file_path` will be used as the source for all documents created from the CSV file.
#
# This is useful when using documents loaded from CSV files for chains that answer questions using sources.

# In[5]:


loader = CSVLoader(file_path="./example_data/mlb_teams_2012.csv", source_column="Team")

data = loader.load()

print(data)


# ## `UnstructuredCSVLoader`
#
# You can also load the table using the `UnstructuredCSVLoader`. One advantage of using `UnstructuredCSVLoader` is that if you use it in `"elements"` mode, an HTML representation of the table will be available in the metadata.

# In[6]:


from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader

loader = UnstructuredCSVLoader(
    file_path="example_data/mlb_teams_2012.csv", mode="elements"
)
docs = loader.load()

print(docs[0].metadata["text_as_html"])


# In[ ]:
