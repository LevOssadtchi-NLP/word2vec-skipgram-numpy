import random
import numpy as np

class SkipGramDataset:
    def __init__(self, encoded_text, window_size=2):
        """
        encoded_text: список индексов слов (выход из Preprocessor.encode)
        window_size: размер контекстного окна вокруг целевого слова
        """
        self.encoded_text = encoded_text
        self.window_size = window_size
        self.pairs = self.generate_pairs()

    def generate_pairs(self):
        """Генерация всех пар (target, context)"""
        pairs = []
        for i, target in enumerate(self.encoded_text):
            start = max(i - self.window_size, 0)
            end = min(i + self.window_size + 1, len(self.encoded_text))
            for j in range(start, end):
                if i != j:
                    context = self.encoded_text[j]
                    pairs.append((target, context))
        return pairs

    def get_batches(self, batch_size=128):
        """Генератор батчей для обучения"""
        random.shuffle(self.pairs)
        for i in range(0, len(self.pairs), batch_size):
            batch = self.pairs[i:i+batch_size]
            targets = np.array([t for t, c in batch], dtype=np.int32)
            contexts = np.array([c for t, c in batch], dtype=np.int32)
            yield targets, contexts

