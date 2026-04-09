import re
from collections import Counter

import numpy as np
from gensim.models import Word2Vec


def is_good_token(tok: str, stopwords: set[str]) -> bool:
    if len(tok) < 3:
        return False
    if tok in stopwords:
        return False
    if re.fullmatch(r'_+', tok):
        return False

    if "_" in tok:
        parts = tok.split("_")
        if len(parts) != 2:
            return False
        if any((not p) or (p in stopwords) or len(p) < 3 for p in parts):
            return False

    return True


def build_word2vec(
    tokenized_with_phrases: list[list[str]],
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    random_state: int,
) -> Word2Vec:
    return Word2Vec(
        sentences=tokenized_with_phrases,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        sg=1,
        epochs=epochs,
        seed=random_state
    )


def build_candidate_matrix(
    tokenized_with_phrases: list[list[str]],
    model: Word2Vec,
    stopwords: set[str],
    candidate_min_freq: int,
) -> tuple[list[str], Counter, np.ndarray]:
    token_counts = Counter(
        tok for tokens in tokenized_with_phrases for tok in tokens
    )

    candidate_tokens = [
        tok for tok, cnt in token_counts.items()
        if cnt >= candidate_min_freq
        and is_good_token(tok, stopwords)
        and tok in model.wv
        and tok.count("_") <= 1
    ]

    if not candidate_tokens:
        raise ValueError("Не удалось отобрать токены для кластеризации.")

    X = np.array([model.wv[tok] for tok in candidate_tokens])
    return candidate_tokens, token_counts, X
