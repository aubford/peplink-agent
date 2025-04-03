# %%
from load.youtube.youtube_load import YoutubeLoad

yt_loader = YoutubeLoad()

# %%
yt_loader.batch_manager.check_batch_and_get_results()
yt_loader.create_synth_data_from_batch_results()

# %%
yt_loader.apply_synth_data_to_staging()

# %% Uncomment to load to vector store
# loader.staging_to_vector_store()
