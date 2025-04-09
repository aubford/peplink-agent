from load.youtube.youtube_load import YoutubeLoad

loader = YoutubeLoad()
if __name__ == "__main__":
    # loader.load()
    # loader.batch_manager.create_batch_job()
    # loader.append_primary_content_embeddings_to_staging_file(chunk_size=200)
    loader.staging_to_vector_store()
