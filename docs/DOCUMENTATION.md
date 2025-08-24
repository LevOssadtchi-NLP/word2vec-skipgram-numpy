# Word2Vec Skip-Gram (NumPy) — Документация

## 1. Обзор проекта

Этот проект реализует алгоритм **Word2Vec (Skip-Gram)** с нуля на чистом **NumPy**, без использования фреймворков глубокого обучения.
Подходит для обучения на произвольных текстах и получения векторных представлений слов.

## 2. Структура проекта

* **`src/`** — исходный код:

  * `train.py` — точка входа для обучения модели.
  * `model.py` — реализация модели Skip-Gram.
  * `dataset.py` — подготовка данных, создание обучающей выборки (окна контекста).
  * `utils.py` — вспомогательные функции (например, токенизация, построение словаря).
* **`data/`** — папка для текстов и обученной модели.
* **`DOCUMENTATION.md`** — этот файл.
* **`README.md`** — краткое описание и скриншоты результатов.

## 3. Установка и запуск

### Требования

* Python 3.9+
* Зависимости:

  ```bash
  pip install numpy tqdm matplotlib
  ```

### Запуск обучения

1. Поместите корпус текста в файл `data/corpus.txt`.
2. Запустите:

   ```bash
   python src/train.py --epochs 10 --embedding_dim 50 --window_size 2 --learning_rate 0.01
   ```
3. Параметры:

   * `--epochs` — число эпох обучения (по умолчанию 10).
   * `--embedding_dim` — размерность векторов слов.
   * `--window_size` — радиус окна контекста.
   * `--learning_rate` — скорость обучения.

### Результат

После обучения:

* Модель сохраняется в `data/embeddings.npy`.
* Словарь (mapping слово→индекс) сохраняется в `data/word2idx.json`.

## 4. Как использовать модель

Пример загрузки эмбеддингов и поиска ближайших слов:

```python
import numpy as np
import json

embeddings = np.load('data/embeddings.npy')
with open('data/word2idx.json') as f:
    word2idx = json.load(f)
idx2word = {i: w for w, i in word2idx.items()}

def most_similar(word, top_n=5):
    if word not in word2idx:
        return []
    vec = embeddings[word2idx[word]]
    sims = embeddings @ vec / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(vec) + 1e-9)
    best = np.argsort(-sims)[:top_n+1]
    return [idx2word[i] for i in best if idx2word[i] != word]

print(most_similar("корпус"))
```

## 5. Архитектура и принципы

* Используется **Skip-Gram с негативным сэмплированием**.
* Потери считаются по функции **cross-entropy**.
* Градиенты обновляются через **SGD**.

Хочешь, я **сразу напишу готовый файл с форматированием и секциями** (чтобы просто вставить)?
Или сделать **ещё более подробную версию с картинками и примерами кода внутри блоков**?
