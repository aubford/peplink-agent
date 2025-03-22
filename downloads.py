import nltk
from spacy.cli import download

"""
Downloads required NLTK datasets for NLP operations.
This centralizes all the NLTK downloads in one place.
"""
nltk.download("brown")
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
download("en_core_web_sm")
download("en_core_web_md")
download("en_core_web_lg")
download("en_core_web_trf")
