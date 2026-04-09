from dataclasses import dataclass, field
import re

RUS_STOPWORDS = {
    "и","в","во","на","с","со","по","к","ко","от","до","за","из","у","о","об",
    "а","но","что","как","для","это","при","также","не","ни","же","ли","или",
    "бы","то","все","весь","этот","тот","такой","какой","который","его","ее",
    "их","мы","вы","они","я","ты","он","она","оно","быть","есть","был","была",
    "были","будет","уже","еще","очень","можно","нужно","только","самый"
}

ENG_STOPWORDS = {
    "the","a","an","and","or","but","if","in","on","at","to","for","of","by",
    "with","from","as","is","are","was","were","be","been","being","this",
    "that","these","those","it","its","their","there","here","about","into",
    "over","after","before","between","out","up","down","off","again","further",
    "then","once","such","no","nor","not","only","own","same","so","than","too",
    "very","can","will","just"
}

ADDITIONAL_RU_STOPWORDS = {
    "если","так","вот","просто","сейчас","вообще","даже",
    "когда","где","здесь","там","потом","поэтому","потому",
    "один","одна","одно","раз","сам","сама","само","сами",
    "наш","ваш","свой","лишь","либо","например","итак"
}

ADDITIONAL_ENG_STOPWORDS = {
    "in_the","on_the","at_the","for_the","to_the","of_the",
    "from_the","by_the","with_the","style_of","based_on",
    "according_to","due_to"
}

TEXT_FIELDS_PRIORITY = [
    "text", "comments_text", "content", "message", "body",
    "post", "caption", "comment", "description"
]

DATE_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"),
    re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}$"),
    re.compile(r"^\d{2}\.\d{2}\.\d{4}$"),
]

@dataclass
class ExplorerConfig:
    min_doc_len: int = 3
    bigram_min_count: int = 3
    bigram_threshold: float = 0.35

    w2v_vector_size: int = 100
    w2v_window: int = 5
    w2v_min_count: int = 3
    w2v_epochs: int = 20

    candidate_min_freq: int = 5
    k_min: int = 3
    k_max: int = 10
    random_state: int = 42

    stopwords: set[str] = field(
        default_factory=lambda: (
            RUS_STOPWORDS
            | ENG_STOPWORDS
            | ADDITIONAL_RU_STOPWORDS
            | ADDITIONAL_ENG_STOPWORDS
        )
    )
