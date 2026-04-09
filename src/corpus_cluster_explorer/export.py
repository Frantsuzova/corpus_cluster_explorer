import json


def save_tokenized_jsonl(
    path: str,
    analysis_label: str,
    selected_fields: list[str],
    records_for_analysis: list[dict],
    tokenized_with_phrases: list[list[str]],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for source_doc, tokens in zip(records_for_analysis, tokenized_with_phrases):
            row = {
                "analysis_label": analysis_label,
                "selected_fields": selected_fields,
                "combined_text": source_doc["combined_text"],
                "field_text_map": source_doc["field_text_map"],
                "tokens": tokens
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_clustered_jsonl(
    path: str,
    analysis_label: str,
    selected_fields: list[str],
    records_for_analysis: list[dict],
    tokenized_with_phrases: list[list[str]],
    candidate_tokens: list[str],
    labels,
    cluster_labels: dict[int, str],
) -> None:
    token_to_cluster = {
        token: int(cluster)
        for token, cluster in zip(candidate_tokens, labels)
    }

    with open(path, "w", encoding="utf-8") as f:
        for source_doc, tokens in zip(records_for_analysis, tokenized_with_phrases):
            doc_cluster_ids = sorted(list({
                token_to_cluster[tok]
                for tok in tokens
                if tok in token_to_cluster
            }))

            row = {
                "analysis_label": analysis_label,
                "selected_fields": selected_fields,
                "combined_text": source_doc["combined_text"],
                "field_text_map": source_doc["field_text_map"],
                "tokens": tokens,
                "cluster_ids": doc_cluster_ids,
                "cluster_labels": [cluster_labels[c] for c in doc_cluster_ids]
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
