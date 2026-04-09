from collections import Counter

from gensim.models.phrases import Phrases, Phraser


def build_bigrams(
    tokenized_texts: list[list[str]],
    min_count: int = 3,
    threshold: float = 0.35,
) -> tuple[list[list[str]], Counter]:
    bigram = Phrases(
        tokenized_texts,
        min_count=min_count,
        threshold=threshold,
        scoring="npmi"
    )
    bigram_phraser = Phraser(bigram)
    tokenized_with_phrases = [bigram_phraser[tokens] for tokens in tokenized_texts]

    phrase_counts = Counter(
        tok for tokens in tokenized_with_phrases for tok in tokens
        if "_" in tok and tok.count("_") == 1
    )

    return tokenized_with_phrases, phrase_counts
