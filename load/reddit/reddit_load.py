# %%
from load.base_load import BaseLoad

loader = BaseLoad("reddit")
loader.load()

# %%

# loader.staging_to_vector_store()
