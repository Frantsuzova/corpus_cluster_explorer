import re

import pymorphy3

CYRILLIC_RE = re.compile(r"^[а-яё]+$")


class TextPreprocessor:
    def __init__(self, stopwords: set[str]) -> None:
        self.stopwords = stopwords
        self.morph = pymorphy3.MorphAnalyzer()

    def normalize_token(self, token: str) -> str:
        if CYRILLIC_RE.fullmatch(token):
            return self.morph.parse(token)[0].normal_form
        return token

    def preprocess(self, text: str) -> list[str]:
        text = text.lower().replace("ё", "е")
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'#\w+', ' ', text)
        text = re.sub(r'[^a-zа-я0-9 ]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = text.split()
        lemmas: list[str] = []

        for token in tokens:
            if len(token) < 3 or token.isdigit():
                continue
            lemma = self.normalize_token(token)
            if lemma and lemma not in self.stopwords:
                lemmas.append(lemma)

        return lemmas
