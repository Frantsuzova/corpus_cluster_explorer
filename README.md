# corpus_cluster_explorer

`corpus_cluster_explorer` — это Python-пакет для загрузки, предобработки и кластеризации **структурированных текстовых датасетов**.

Пакет поддерживает два основных сценария работы:

1. **Полный пайплайн**  
   Загрузка структурированного датасета, автоматическое обнаружение текстовых полей, выбор полей для анализа, токенизация корпуса, извлечение биграмм, обучение Word2Vec, оценка качества кластеризации, запуск KMeans и экспорт результатов.

2. **Продолжение работы с токенизированного корпуса**  
   Загрузка ранее сохранённого токенизированного корпуса в формате JSONL и продолжение анализа сразу с этапа эмбеддингов и кластеризации.

## Возможности

- загрузка структурированных текстовых датасетов:
  - JSONL
  - JSON
  - CSV
  - TSV
- автоматическое обнаружение текстовых полей
- предобработка текста:
  - приведение к нижнему регистру
  - удаление ссылок
  - удаление `@username`
  - удаление хэштегов
  - очистка пунктуации
  - лемматизация русских слов с помощью `pymorphy3`
- извлечение биграмм с помощью `gensim.Phrases`
- обучение Word2Vec
- кластеризация с помощью KMeans
- оценка качества кластеризации через silhouette score
- PCA-проекция для визуализации
- экспорт:
  - токенизированного корпуса
  - корпуса с кластерной разметкой

## Установка

### Установка из GitHub

```bash
pip install git+https://github.com/YOUR_USERNAME/corpus_cluster_explorer.git
```

### Локальная editable-установка

```bash
pip install -e .
```

## Быстрый старт

### 1. Полный пайплайн

```python
from corpus_cluster_explorer import CorpusExplorer

explorer = CorpusExplorer()

explorer.load("posts.jsonl")
print(explorer.text_fields)

explorer.choose_fields(["text", "comments_text"])
explorer.tokenize()
explorer.save_tokenized("tokenized.jsonl")

explorer.fit_embeddings()
explorer.evaluate_clusters()
explorer.cluster(4)

explorer.save_clustered("clustered.jsonl")
```

### 2. Продолжение работы с токенизированного корпуса

```python
from corpus_cluster_explorer import CorpusExplorer

explorer = CorpusExplorer()

explorer.load_tokenized("tokenized.jsonl")

explorer.fit_embeddings()
explorer.evaluate_clusters()
explorer.cluster(4)

explorer.save_clustered("clustered.jsonl")
```

## Обзор API

### Загрузка

```python
explorer.load("data.jsonl")
explorer.text_fields
```

### Выбор полей

```python
explorer.choose_fields(["text", "comments_text"])
```

### Токенизация

```python
explorer.tokenize()
```

### Статистика

```python
explorer.token_stats()
```

### Кластеризация

```python
explorer.fit_embeddings()
explorer.evaluate_clusters()
explorer.cluster(4)
```

### Сохранение

```python
explorer.save_tokenized("tokenized.jsonl")
explorer.save_clustered("clustered.jsonl")
```

## CLI

```bash
corpus-explorer data.jsonl --fields text comments_text --clusters 4
```

## Форматы

**Tokenized JSONL**:
- tokens
- combined_text
- field_text_map

**Clustered JSONL**:
- tokens
- cluster_ids
- cluster_labels

## Лицензия

MIT
