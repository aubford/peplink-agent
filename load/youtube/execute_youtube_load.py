from load.youtube.youtube_load import YouTubeLoad

if __name__ == "__main__":
    loader = YouTubeLoad()
    loader.load_from_merged()
    # Uncomment to load to vector store
    # loader.staging_to_vector_store()
