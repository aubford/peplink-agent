# %%
from load.reddit_general.reddit_general_load import RedditGeneralLoad

rg_loader = RedditGeneralLoad()
# %%
rg_loader.batch_manager.check_batch_and_get_results()
# %%
rg_loader.create_synth_data_from_batch_results()
rg_loader.apply_synth_data_to_staging()

# %% Uncomment to load to vector store
# loader.staging_to_vector_store()
