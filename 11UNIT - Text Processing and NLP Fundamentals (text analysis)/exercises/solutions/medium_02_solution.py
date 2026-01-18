"""Solutions for medium_02_text_normalisation.py"""
import re
import unicodedata

def lowercase_text(text: str) -> str:
    return text.lower()

def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text)

def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", "", text)

def normalise_unicode(text: str, form: str = "NFC") -> str:
    return unicodedata.normalize(form, text)

def remove_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def remove_stopwords(tokens: list[str], stopwords: set[str]) -> list[str]:
    return [t for t in tokens if t.lower() not in stopwords]

def create_pipeline(*functions):
    def pipeline(text: str) -> str:
        result = text
        for func in functions:
            result = func(result)
        return result
    return pipeline
