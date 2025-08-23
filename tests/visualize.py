import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# --- 1. Загружаем CSV ---
# CSV: первый столбец - слово, следующие 50 - координаты эмбеддинга
file_path = "../src/artifacts/embeddings.csv"
df = pd.read_csv(file_path, header=None)[:1000]

# Разделяем слова и векторы
words = df.iloc[:, 0].values
vectors = df.iloc[:, 1:].values

# --- 2. Уменьшаем размерность до 2D (t-SNE) ---
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
vectors_2d = tsne.fit_transform(vectors)

# --- 3. Визуализация ---
plt.figure(figsize=(12, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=10, alpha=0.7)

# Подписываем только часть слов, чтобы не было слишком много текста
for i, word in enumerate(words):
    plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=8, alpha=0.7)

plt.title("Визуализация эмбеддингов (t-SNE)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
