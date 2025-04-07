#!/usr/bin/env python
# coding: utf-8

# # Geopandas
# 
# [Geopandas](https://geopandas.org/en/stable/index.html) is an open-source project to make working with geospatial data in python easier. 
# 
# GeoPandas extends the datatypes used by pandas to allow spatial operations on geometric types. 
# 
# Geometric operations are performed by shapely. Geopandas further depends on fiona for file access and matplotlib for plotting.
# 
# LLM applications (chat, QA) that utilize geospatial data are an interesting area for exploration.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  sodapy')
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  pandas')
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  geopandas')


# In[2]:


import ast

import geopandas as gpd
import pandas as pd
from langchain_community.document_loaders import OpenCityDataLoader


# Create a GeoPandas dataframe from [`Open City Data`](/docs/integrations/document_loaders/open_city_data) as an example input.

# In[ ]:


# Load Open City Data
dataset = "tmnf-yvry"  # San Francisco crime data
loader = OpenCityDataLoader(city_id="data.sfgov.org", dataset_id=dataset, limit=5000)
docs = loader.load()


# In[30]:


# Convert list of dictionaries to DataFrame
df = pd.DataFrame([ast.literal_eval(d.page_content) for d in docs])

# Extract latitude and longitude
df["Latitude"] = df["location"].apply(lambda loc: loc["coordinates"][1])
df["Longitude"] = df["location"].apply(lambda loc: loc["coordinates"][0])

# Create geopandas DF
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
)

# Only keep valid longitudes and latitudes for San Francisco
gdf = gdf[
    (gdf["Longitude"] >= -123.173825)
    & (gdf["Longitude"] <= -122.281780)
    & (gdf["Latitude"] >= 37.623983)
    & (gdf["Latitude"] <= 37.929824)
]


# Visualization of the sample of SF crime data. 

# In[ ]:


import matplotlib.pyplot as plt

# Load San Francisco map data
sf = gpd.read_file("https://data.sfgov.org/resource/3psu-pn9h.geojson")

# Plot the San Francisco map and the points
fig, ax = plt.subplots(figsize=(10, 10))
sf.plot(ax=ax, color="white", edgecolor="black")
gdf.plot(ax=ax, color="red", markersize=5)
plt.show()


# Load GeoPandas dataframe as a `Document` for downstream processing (embedding, chat, etc). 
# 
# The `geometry` will be the default `page_content` columns, and all other columns are placed in `metadata`.
# 
# But, we can specify the `page_content_column`.

# In[32]:


from langchain_community.document_loaders import GeoDataFrameLoader

loader = GeoDataFrameLoader(data_frame=gdf, page_content_column="geometry")
docs = loader.load()


# In[33]:


docs[0]

