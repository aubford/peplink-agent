import json
import pandas as pd
from pathlib import Path
from transform.base_transform import BaseTransform

class RedditTransform(BaseTransform):
    def __init__(self):
        super().__init__("reddit")

    def transform_file(self, file_path: Path) -> pd.DataFrame:
        with open(file_path, 'r') as f:
            data = json.load(f)

        posts = []
        for post in data:
            author = post['metadata']['post_author']
            transformed_post = {
                # Main post content
                'id': post['metadata']['post_id'],
                'page_content': post['page_content'],

                # Post metadata
                'subreddit': post['metadata']['post_subreddit'],
                'category': post['metadata']['post_category'],
                'title': post['metadata']['post_title'],
                'score': post['metadata']['post_score'],
                'url': post['metadata']['post_url'],

                # Author metadata
                'author_name': author.get('name', ''),
                'author_id': author.get('id', ''),
                'author_is_employee': author.get('is_employee', False),
                'author_is_mod': author.get('is_mod', False),
                'author_is_gold': author.get('is_gold', False),
                'author_verified': author.get('verified', False),
                'author_has_verified_email': author.get('has_verified_email', False),
                'author_hide_from_robots': author.get('hide_from_robots', False),
                'author_is_blocked': author.get('is_blocked', False),
                'author_accept_followers': author.get('accept_followers', True),
                'author_has_subscribed': author.get('has_subscribed', True),

                # Author karma
                'author_total_karma': author.get('total_karma', 0),
                'author_awardee_karma': author.get('awardee_karma', 0),
                'author_awarder_karma': author.get('awarder_karma', 0),
                'author_link_karma': author.get('link_karma', 0),
                'author_comment_karma': author.get('comment_karma', 0),

                # Source tracking
                'source_file': file_path.name
            }

            posts.append(transformed_post)

        df = pd.DataFrame(posts)

        # Convert numeric fields
        # numeric_fields = [
        #     'score', 'author_total_karma', 'author_awardee_karma', 'author_awarder_karma',
        #     'author_link_karma', 'author_comment_karma'
        # ]
        # for field in numeric_fields:
        #     df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0).astype('Int64')

        # Convert boolean fields
        # boolean_fields = [
        #     'author_is_employee', 'author_is_mod', 'author_is_gold', 'author_verified',
        #     'author_has_verified_email', 'author_hide_from_robots', 'author_is_blocked',
        #     'author_accept_followers', 'author_has_subscribed'
        # ]
        # for field in boolean_fields:
        #     df[field] = df[field].fillna(False).astype(bool)

        return df
