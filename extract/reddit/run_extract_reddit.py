#%%
from extract.reddit.reddit_extractor import RedditExtractor


pep_extractor = RedditExtractor('Peplink')
pep_extractor.extract()

#%%
from extract.reddit.reddit_extractor import RedditExtractor

net_extractor = RedditExtractor('networking')
net_extractor.extract()

#%%
from extract.reddit.reddit_extractor import RedditExtractor

sys_extractor = RedditExtractor('sysadmin')
sys_extractor.extract()
