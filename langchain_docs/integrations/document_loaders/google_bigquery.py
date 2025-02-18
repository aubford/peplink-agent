#!/usr/bin/env python
# coding: utf-8

# # Google BigQuery
#
# >[Google BigQuery](https://cloud.google.com/bigquery) is a serverless and cost-effective enterprise data warehouse that works across clouds and scales with your data.
# `BigQuery` is a part of the `Google Cloud Platform`.
#
# Load a `BigQuery` query with one document per row.

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet langchain-google-community[bigquery]"
)


# In[3]:


from langchain_google_community import BigQueryLoader


# In[3]:


BASE_QUERY = """
SELECT
  id,
  dna_sequence,
  organism
FROM (
  SELECT
    ARRAY (
    SELECT
      AS STRUCT 1 AS id, "ATTCGA" AS dna_sequence, "Lokiarchaeum sp. (strain GC14_75)." AS organism
    UNION ALL
    SELECT
      AS STRUCT 2 AS id, "AGGCGA" AS dna_sequence, "Heimdallarchaeota archaeon (strain LC_2)." AS organism
    UNION ALL
    SELECT
      AS STRUCT 3 AS id, "TCCGGA" AS dna_sequence, "Acidianus hospitalis (strain W1)." AS organism) AS new_array),
  UNNEST(new_array)
"""


# ## Basic Usage

# In[6]:


loader = BigQueryLoader(BASE_QUERY)

data = loader.load()


# In[7]:


print(data)


# ## Specifying Which Columns are Content vs Metadata

# In[8]:


loader = BigQueryLoader(
    BASE_QUERY,
    page_content_columns=["dna_sequence", "organism"],
    metadata_columns=["id"],
)

data = loader.load()


# In[9]:


print(data)


# ## Adding Source to Metadata

# In[18]:


# Note that the `id` column is being returned twice, with one instance aliased as `source`
ALIASED_QUERY = """
SELECT
  id,
  dna_sequence,
  organism,
  id as source
FROM (
  SELECT
    ARRAY (
    SELECT
      AS STRUCT 1 AS id, "ATTCGA" AS dna_sequence, "Lokiarchaeum sp. (strain GC14_75)." AS organism
    UNION ALL
    SELECT
      AS STRUCT 2 AS id, "AGGCGA" AS dna_sequence, "Heimdallarchaeota archaeon (strain LC_2)." AS organism
    UNION ALL
    SELECT
      AS STRUCT 3 AS id, "TCCGGA" AS dna_sequence, "Acidianus hospitalis (strain W1)." AS organism) AS new_array),
  UNNEST(new_array)
"""


# In[19]:


loader = BigQueryLoader(ALIASED_QUERY, metadata_columns=["source"])

data = loader.load()


# In[20]:


print(data)
