from typing import List
from extract.base_extractor import BaseExtractor


class ExtractorManager:
    """Manager for handling multiple extractors of the same type for use in notebook execution."""

    def __init__(self, extractors: List[BaseExtractor]):
        """
        Initialize manager with extractors.
        Args:
            extractors: List of configured BaseExtractor instances
        """
        self.extractors = extractors

    def fetch_all(self) -> None:
        """Extract data from all sources and remove completed extractors."""
        extractors = self.extractors.copy()
        for extractor in extractors:
            extractor.extract()
            self.extractors.remove(extractor)
