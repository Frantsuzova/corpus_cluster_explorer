from .clustering import evaluate_k_range, project_pca, run_kmeans
from .config import ExplorerConfig
from .embeddings import build_candidate_matrix, build_word2vec
from .export import save_clustered_jsonl, save_tokenized_jsonl
from .io import (
    build_analysis_documents,
    detect_text_fields,
    load_structured_dataset,
    load_tokenized_jsonl,
)
from .phrases import build_bigrams
from .preprocessing import TextPreprocessor


class CorpusExplorer:
    def __init__(self, config: ExplorerConfig | None = None) -> None:
        self.config = config or ExplorerConfig()
        self.preprocessor = TextPreprocessor(self.config.stopwords)

        self.records = None
        self.text_fields = None

        self.selected_fields = None
        self.analysis_label = None
        self.analysis_docs = None
        self.analysis_texts = None
        self.records_for_analysis = None

        self.tokenized_texts = None
        self.tokenized_with_phrases = None
        self.phrase_counts = None

        self.w2v_model = None
        self.token_counts = None
        self.candidate_tokens = None
        self.X = None

        self.valid_k = None
        self.scores = None

        self.df_clusters = None
        self.labels = None
        self.cluster_labels = None

    # -----------------------------------------------------
    # Режим 1: старт со структурированного датасета
    # -----------------------------------------------------
    def load(self, path: str) -> list[dict]:
        self.records = load_structured_dataset(path)
        self.text_fields = detect_text_fields(self.records)
        return self.records

    def choose_fields(self, selected_fields: list[str]) -> None:
        if self.records is None:
            raise ValueError("Сначала загрузите датасет через .load().")

        self.selected_fields = selected_fields
        self.analysis_label = " + ".join(selected_fields)

        self.analysis_docs = build_analysis_documents(self.records, selected_fields)
        self.analysis_texts = [
            doc["combined_text"] for doc in self.analysis_docs
            if doc["combined_text"].strip()
        ]
        self.records_for_analysis = [
            doc for doc in self.analysis_docs
            if doc["combined_text"].strip()
        ]

    def tokenize(self) -> None:
        if self.analysis_texts is None:
            raise ValueError("Сначала выберите поля через .choose_fields().")

        self.tokenized_texts = [
            self.preprocessor.preprocess(text)
            for text in self.analysis_texts
        ]
        self.tokenized_texts = [
            tokens for tokens in self.tokenized_texts
            if len(tokens) >= self.config.min_doc_len
        ]

        self.tokenized_with_phrases, self.phrase_counts = build_bigrams(
            self.tokenized_texts,
            min_count=self.config.bigram_min_count,
            threshold=self.config.bigram_threshold,
        )

    # -----------------------------------------------------
    # Режим 2: старт с уже токенизированного корпуса
    # -----------------------------------------------------
    def load_tokenized(self, path: str) -> list[dict]:
        rows = load_tokenized_jsonl(path)

        self.records_for_analysis = []
        self.tokenized_with_phrases = []

        for row in rows:
            tokens = row["tokens"]
            if not isinstance(tokens, list):
                raise ValueError("Поле 'tokens' должно быть списком.")
            self.tokenized_with_phrases.append(tokens)
            self.records_for_analysis.append({
                "combined_text": row.get("combined_text", ""),
                "field_text_map": row.get("field_text_map", {})
            })

        self.analysis_label = rows[0].get("analysis_label", "tokenized_corpus") if rows else "tokenized_corpus"
        self.selected_fields = rows[0].get("selected_fields", []) if rows else []

        return rows

    # -----------------------------------------------------
    # Эмбеддинги и кластеризация
    # -----------------------------------------------------
    def fit_embeddings(self) -> None:
        if self.tokenized_with_phrases is None:
            raise ValueError("Нет токенизированного корпуса. Выполните .tokenize() или .load_tokenized().")

        self.w2v_model = build_word2vec(
            self.tokenized_with_phrases,
            vector_size=self.config.w2v_vector_size,
            window=self.config.w2v_window,
            min_count=self.config.w2v_min_count,
            epochs=self.config.w2v_epochs,
            random_state=self.config.random_state,
        )

        self.candidate_tokens, self.token_counts, self.X = build_candidate_matrix(
            self.tokenized_with_phrases,
            self.w2v_model,
            self.config.stopwords,
            self.config.candidate_min_freq,
        )

    def evaluate_clusters(self) -> tuple[list[int], list[float]]:
        if self.X is None:
            raise ValueError("Сначала вызовите .fit_embeddings().")

        self.valid_k, self.scores = evaluate_k_range(
            self.X,
            self.config.k_min,
            self.config.k_max,
            self.config.random_state,
        )
        return self.valid_k, self.scores

    def cluster(self, n_clusters: int) -> None:
        if self.X is None or self.w2v_model is None:
            raise ValueError("Сначала вызовите .fit_embeddings().")

        self.df_clusters, self.labels, self.cluster_labels = run_kmeans(
            self.X,
            self.candidate_tokens,
            self.token_counts,
            self.w2v_model,
            n_clusters,
            self.config.random_state,
        )

    def pca(self):
        if self.X is None:
            raise ValueError("Сначала вызовите .fit_embeddings().")
        return project_pca(self.X, self.config.random_state)

    # -----------------------------------------------------
    # Экспорт
    # -----------------------------------------------------
    def save_tokenized(self, path: str) -> None:
        if self.tokenized_with_phrases is None:
            raise ValueError("Нет токенизированного корпуса для сохранения.")

        save_tokenized_jsonl(
            path,
            self.analysis_label or "unknown",
            self.selected_fields or [],
            self.records_for_analysis or [],
            self.tokenized_with_phrases,
        )

    def save_clustered(self, path: str) -> None:
        if self.labels is None or self.cluster_labels is None:
            raise ValueError("Сначала выполните .cluster().")

        save_clustered_jsonl(
            path,
            self.analysis_label or "unknown",
            self.selected_fields or [],
            self.records_for_analysis or [],
            self.tokenized_with_phrases or [],
            self.candidate_tokens or [],
            self.labels,
            self.cluster_labels,
        )

    # -----------------------------------------------------
    # Отчётные данные
    # -----------------------------------------------------
    def token_stats(self) -> dict:
        if self.tokenized_with_phrases is None:
            raise ValueError("Нет токенизированного корпуса.")

        total_tokens = sum(len(doc) for doc in self.tokenized_with_phrases)
        unique_tokens = len(set(tok for doc in self.tokenized_with_phrases for tok in doc))

        return {
            "documents": len(self.tokenized_with_phrases),
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "top_phrases": self.phrase_counts.most_common(10) if self.phrase_counts else [],
            "sample_document": self.tokenized_with_phrases[0] if self.tokenized_with_phrases else [],
        }
