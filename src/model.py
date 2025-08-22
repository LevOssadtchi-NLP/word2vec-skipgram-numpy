import numpy as np
from tqdm import tqdm

class SkipGram:
    def __init__(self, vocab_size, embedding_dim=50, learning_rate=0.01, neg_sampling=False, neg_samples=5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        self.neg_sampling = neg_sampling
        self.neg_samples = neg_samples

        # W: вектор слов, W': вектор контекстов
        self.W = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_prime = np.random.randn(embedding_dim, vocab_size) * 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, target_idx):
        """Прямой проход для одного target"""
        h = self.W[target_idx]  # (embedding_dim,)
        return h

    def backward(self, h, context_idx):
        """Обратное распространение с опцией negative sampling"""
        if self.neg_sampling:
            # Случайные негативные примеры
            neg_idx = np.random.choice(
                [i for i in range(self.vocab_size) if i != context_idx],
                self.neg_samples,
                replace=False
            )
            indices = np.array([context_idx] + list(neg_idx))
            labels = np.array([1] + [0]*self.neg_samples)  # positive = 1, negative = 0
        else:
            # Полный softmax
            indices = np.arange(self.vocab_size)
            labels = np.zeros(self.vocab_size)
            labels[context_idx] = 1

        # предсказания
        u = np.dot(h, self.W_prime[:, indices])  # (len(indices),)
        y_pred = self.sigmoid(u)

        # ===== Loss =====
        eps = 1e-10  # чтобы не было log(0)
        loss = -np.sum(labels * np.log(y_pred + eps) + (1 - labels) * np.log(1 - y_pred + eps))

        # ошибка
        e = y_pred - labels  # (len(indices),)

        # градиенты
        dW_prime = np.outer(h, e)                 # (embedding_dim, len(indices))
        dW = np.dot(self.W_prime[:, indices], e)  # (embedding_dim,)

        # обновление весов
        self.W_prime[:, indices] -= self.lr * dW_prime

        return dW, loss

    def train(self, dataset, epochs=1):
        """Обучение на всем датасете"""
        for epoch in range(epochs):
            total_loss = 0
            num_samples = 0

            print(f"Epoch {epoch+1}/{epochs}")
            for i, (targets, contexts) in tqdm(enumerate(dataset.get_batches(batch_size=128))):
                for t, c in zip(targets, contexts):
                    h = self.forward(t)
                    dW, loss = self.backward(h, c)
                    self.W[t] -= self.lr * dW

                    total_loss += loss
                    num_samples += 1

            avg_loss = total_loss / num_samples
            print(f"Done epoch {epoch+1}, loss = {avg_loss:.4f}")

    def get_embedding(self, word_idx):
        """Получить вектор слова"""
        return self.W[word_idx]
