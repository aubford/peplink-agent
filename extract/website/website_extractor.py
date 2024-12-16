# %%

from IPython.display import display
from langchain_community.document_loaders import RecursiveUrlLoader
import time
from bs4 import BeautifulSoup
import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def extractor(html_content: str) -> str:
    """Custom extractor for HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove unwanted elements
    for element in soup.find_all(['nav', 'footer', 'script', 'style']):
        element.decompose()

    # Extract main content
    return soup.get_text(separator=' ', strip=True)


loader = RecursiveUrlLoader(
    url="https://www.peplink.com",
    base_url="https://www.peplink.com",
    use_async=False,
    max_depth=3,
    extractor=extractor,
    prevent_outside=True,
    headers=headers,
    exclude_dirs=(
        "/cart",
        "/login",
        "/account",
        "/search",
        "/pdf",
        ".pdf",
        "/wp-admin/",
        "/wp-includes/",
        "/cgi-bin/",
        "/private/",
        "/tmp/",
        "/admin/",
        "/login/",
        "/user/",
        "/dashboard/",
        "/scripts/",
        "/styles/"
    )
)

docs = []
for doc in loader.lazy_load():
    docs.append(doc)
    time.sleep(1)
