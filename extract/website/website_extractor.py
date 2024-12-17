#%%

from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup
from extract.base_extractor import BaseExtractor, Ldoc
from util.util import serialize_document
import re

class WebsiteExtractor(BaseExtractor):
    def __init__(self, url: str, *, base_url: str = None, max_depth: int = 6):
        super().__init__("web")
        self.file_id = url.split("://", 1)[-1].replace("www.", "").replace("/", "_")[:40]

        self.loader = RecursiveUrlLoader(
            url=url,
            base_url=base_url,
            use_async=False,
            max_depth=max_depth,
            extractor=self._extract_content,
            link_regex=self._build_link_regex(),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }

        )

    @staticmethod
    def _build_link_regex() -> str:
        """Build regex pattern for filtering links during crawling."""
        # Email pattern
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

        # Prefixes to ignore
        prefixes = ("javascript:", "mailto:", "#", f"(?:{email_pattern})", "/cdn-cgi/")
        prefixes_regex = "(?!" + "|".join([re.escape(s) for s in prefixes[:4]]) + f"|{prefixes[4]})"

        # Suffixes to ignore
        suffixes = (".css", ".js", ".ico", ".png", ".jpg", ".jpeg", ".gif", ".svg",
                    ".csv", ".bz2", ".zip", ".epub")
        suffixes_regex = "(?!" + "|".join([re.escape(s) + r"[\#'\"]" for s in suffixes]) + ")"

        return rf"href=[\"']{prefixes_regex}((?:{suffixes_regex}.)*?)[\#'\"]"

    @staticmethod
    def _extract_content(html_content: str) -> str:
        """Extract clean text content from HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove unwanted elements
        for element in soup.find_all(['nav', 'footer', 'script', 'style']):
            element.decompose()

        return soup.get_text(separator=' ', strip=True)

    def extract(self):
        """Extract and process website content"""
        stream_key = self.start_stream(Ldoc, identifier=self.file_id)
        for doc in self.loader.lazy_load():
            self.stream_item(serialize_document(doc), stream_key)
        self.end_stream(stream_key)


extractor = WebsiteExtractor("https://www.javatpoint.com/fundamentals-of-computer-networking", base_url="https://www.javatpoint.com/")
extractor.extract()
