# %%
from load.reddit_general.reddit_general_load import RedditGeneralLoad

loader = RedditGeneralLoad()

# %% Check batch job status
loader.batch_manager.check_batch_and_get_results()

# %% Create synth data and apply to staging
loader.create_synth_data_from_batch_results()

# %%
loader.apply_synth_data_to_staging()

# %% Uncomment to load to vector store
# loader.staging_to_vector_store()
