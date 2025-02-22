from load.youtube.youtube_load import YoutubeLoad

if __name__ == "__main__":
    loader = YoutubeLoad()
    loader.load()
    # Uncomment to load to vector store
    # loader.staging_to_vector_store()
