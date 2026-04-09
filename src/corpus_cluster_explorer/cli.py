import argparse

import matplotlib.pyplot as plt

from .pipeline import CorpusExplorer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Corpus explorer for structured text datasets"
    )

    parser.add_argument("path", help="Path to structured dataset or tokenized JSONL")
    parser.add_argument(
        "--mode",
        choices=["structured", "tokenized"],
        default="structured",
        help="Input mode"
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        help="Fields to analyze in structured mode"
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help="Number of clusters"
    )
    parser.add_argument(
        "--save-tokenized",
        help="Path to save tokenized JSONL"
    )
    parser.add_argument(
        "--save-clustered",
        help="Path to save clustered JSONL"
    )

    args = parser.parse_args()

    explorer = CorpusExplorer()

    if args.mode == "structured":
        explorer.load(args.path)
        print(f"Размер корпуса: {len(explorer.records)} документов")
        print("Обнаруженные текстовые поля:", explorer.text_fields)

        fields = args.fields or explorer.text_fields
        explorer.choose_fields(fields)
        explorer.tokenize()

        stats = explorer.token_stats()
        print(f"\nАнализируются данные: {' + '.join(fields)}")
        print(f"Размер анализируемого корпуса: {len(explorer.analysis_texts)} документов")
        print(f"Всего токенов после токенизации: {stats['total_tokens']}")
        print(f"Уникальных токенов: {stats['unique_tokens']}")
        print("\nТоп-10 найденных фразовых токенов:")
        for phrase, count in stats["top_phrases"]:
            print(f"{phrase:<35} {count}")

        if args.save_tokenized:
            explorer.save_tokenized(args.save_tokenized)
            print(f"\nТокенизированный корпус сохранён: {args.save_tokenized}")

    else:
        explorer.load_tokenized(args.path)
        stats = explorer.token_stats()
        print("Загружен токенизированный корпус.")
        print(f"Документов: {stats['documents']}")
        print(f"Всего токенов: {stats['total_tokens']}")
        print(f"Уникальных токенов: {stats['unique_tokens']}")

    explorer.fit_embeddings()
    valid_k, scores = explorer.evaluate_clusters()

    print("\nОценка кластеров:")
    for k, score in zip(valid_k, scores):
        print(f"k={k}: silhouette={score:.4f}")

    best_k = valid_k[scores.index(max(scores))]
    n_clusters = args.clusters or best_k
    print(f"\nИспользуется {n_clusters} кластеров")

    explorer.cluster(n_clusters)

    for cluster_id in range(n_clusters):
        cluster_df = explorer.df_clusters[explorer.df_clusters["cluster"] == cluster_id].copy()
        top_freq = cluster_df.sort_values("freq", ascending=False).head(10)
        top_center = cluster_df.sort_values("dist_to_center", ascending=True).head(10)

        print("\n" + "=" * 90)
        print(f"КЛАСТЕР {cluster_id}")
        print("=" * 90)
        print("\nСамые частотные токены:")
        print(", ".join(top_freq["token"].tolist()))
        print("\nСамые типичные токены:")
        print(", ".join(top_center["token"].tolist()))

    X_2d = explorer.pca()
    plt.figure(figsize=(12, 8))
    for cluster_id in range(n_clusters):
        idx = explorer.labels == cluster_id
        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            label=f"Cluster {cluster_id}",
            alpha=0.7
        )
    plt.title("Визуализация кластеров (PCA)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    if args.save_clustered:
        explorer.save_clustered(args.save_clustered)
        print(f"\nКорпус с кластерной разметкой сохранён: {args.save_clustered}")
