# corpus_cluster_explorer

`corpus_cluster_explorer` — это Python-пакет для загрузки, предобработки и кластеризации **структурированных текстовых датасетов**.

Пакет поддерживает два основных сценария работы:

1. **Полный пайплайн**  
   Загрузка структурированного датасета, автоматическое обнаружение текстовых полей, выбор полей для анализа, токенизация корпуса, извлечение биграмм, обучение Word2Vec, оценка качества кластеризации, запуск KMeans и экспорт результатов.

2. **Продолжение работы с токенизированного корпуса**  
   Загрузка ранее сохранённого токенизированного корпуса в формате JSONL и продолжение анализа сразу с этапа эмбеддингов и кластеризации.

---

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
- извлечение биграмм (`gensim.Phrases`)
- обучение Word2Vec
- кластеризация (KMeans)
- оценка качества (silhouette score)
- PCA для визуализации
- экспорт:
  - токенизированного корпуса
  - корпуса с кластерной разметкой

---

## Установка

```bash
pip install corpus-cluster-explorer
```

---

## Быстрый старт

### Полный пайплайн

```python
from corpus_cluster_explorer import CorpusExplorer

explorer = CorpusExplorer()

# загрузка данных
explorer.load("data.jsonl")

# посмотреть какие текстовые поля найдены
print(explorer.text_fields)

# выбрать поля
explorer.choose_fields(explorer.text_fields)

# токенизация
explorer.tokenize()

# сохранить при необходимости
explorer.save_tokenized("tokenized.jsonl")

# эмбеддинги + подбор кластеров
explorer.fit_embeddings()
valid_k, scores = explorer.evaluate_clusters()
print(valid_k, scores)

# кластеризация
explorer.cluster(4)

# сохранить результат
explorer.save_clustered("clustered.jsonl")
```

---

### Продолжение с токенизированного корпуса

```python
from corpus_cluster_explorer import CorpusExplorer

explorer = CorpusExplorer()

explorer.load_tokenized("tokenized.jsonl")

explorer.fit_embeddings()
valid_k, scores = explorer.evaluate_clusters()

explorer.cluster(4)
explorer.save_clustered("clustered.jsonl")
```

---

## API

### Загрузка

```python
explorer.load("data.jsonl")
explorer.text_fields
```

### Выбор полей

```python
explorer.choose_fields(["text"])
```

или

```python
explorer.choose_fields(explorer.text_fields)
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

---

## CLI

```bash
corpus-explorer data.jsonl --fields text comments_text --clusters 4
```

---

## Форматы данных

### Tokenized JSONL
- tokens  
- combined_text  
- field_text_map  

### Clustered JSONL
- tokens  
- cluster_ids  
- cluster_labels  

---

## Замечания

- русские слова лемматизируются  
- нерусские токены сохраняются  
- используются только **биграммы**  
- можно продолжать работу с сохранённого токенизированного корпуса  
- рекомендуется сначала посмотреть `explorer.text_fields`, затем выбирать поля  

---

## Лицензия

MIT
