from load.reddit_general.reddit_general_load import RedditGeneralLoad

loader = RedditGeneralLoad()
if __name__ == "__main__":
    loader.load()
    # loader.batch_manager.create_batch_job()
