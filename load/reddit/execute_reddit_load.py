from load.reddit.reddit_load import RedditLoad

loader = RedditLoad()
if __name__ == "__main__":
    loader.load()
    # loader.batch_manager.create_batch_job()
