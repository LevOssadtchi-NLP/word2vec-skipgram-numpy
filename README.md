# word2vec-skipgram-numpy
Simple NumPy implementation of Word2Vec (skip-gram) for learning word embeddings from scratch.
Простой и понятный с нуля реализованный алгоритм **Word2Vec (skip-gram)** на базе `NumPy`.  
Проект показывает, как можно обучить эмбеддинги слов без готовых фреймворков (PyTorch, TensorFlow).

## Возможности
- Очистка и токенизация корпуса
- Подготовка датасета для skip-gram
- Обучение skip-gram модели (negative sampling optional)
- Получение векторов слов
- Поиск похожих слов по косинусному сходству

## Пример
```python
from src.model import SkipGram
from src.utils import most_similar

model = SkipGram(vocab_size=5000, embedding_dim=50)
model.train(corpus="data/ruwiki_sample.txt", epochs=3, learning_rate=0.01)

print(most_similar("москва", model, top_n=5))
