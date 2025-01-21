# %%

from transform.html.html_transform import HTMLTransform
from IPython.display import display

dfs = HTMLTransform.get_parquet_dfs()
df = dfs[0]

display(df.shape)
display(df.head())
display(df.describe())
display(df.info())
display(df.columns)

# %%

# Show rows where section ends with '#'
mask = df["section"].str.endswith("#", na=False)
print("\nRows where section ends with '#':")
print(f"Found {mask.sum()} rows")
if mask.any():
    display(df[mask][["section", "page_content"]])


# %%

# Find duplicate page content and show counts
duplicates_df = df[df.duplicated(subset=["page_content"], keep=False)].sort_values("page_content")
print(f"\nNumber of rows with duplicate content: {len(duplicates_df)}")
print("\nBreakdown of duplicates:")
print(duplicates_df["page_content"].value_counts())


# %%

# Display top 10 page contents
print("\nTop 10 page contents:")
for i, content in enumerate(df["page_content"].head(10), 1):
    print(f"\n{i}. {'-'*80}\n{content}")

# %%

import re


def is_product_list(text: str) -> bool:
    """Check if text appears to be a product listing."""
    # Check for repeated product prefix
    if text.count("MAX ") <= 30:  # Multiple occurrences of "MAX" suggest a product list
        return False

    if text.count("(CAT-") < 3:
        return False

    if len([m for m in re.findall(r"BR.", text)]) < 3:  # Multiple occurrences of BR product codes
        return False

    # Check for many uppercase words (typical in product names)
    uppercase_words = sum(1 for word in text.split() if word.isupper())
    if uppercase_words / len(text.split()) < 0.2:  # More than 30% words are uppercase
        return False

    return True


# Filter out product listings
clean_df = df[~df["page_content"].apply(is_product_list)]

print(f"\nRows removed: {len(df) - len(clean_df)}")
print(f"Rows remaining: {len(clean_df)}")


# contents = df["page_content"].head(10)
# for content in contents:
#     print(content)
#     print("-" * 100)


# Sample some filtered rows to verify
# print("\nSample of filtered out content:")
# filtered = df[df["page_content"].apply(is_product_list)].sample(n=5, random_state=42)
# display(filtered)
