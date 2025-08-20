import re
from collections import Counter

class Preprocessor:
    def __init__(self, vocab_size=10000, lowercase=True):
        """
        vocab_size: максимальное количество слов в словаре
        lowercase: приводить ли слова к нижнему регистру
        """
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.word2idx = {}
        self.idx2word = {}

    def clean_text(self, text: str) -> str:
        """Простейшая очистка текста: убираем лишние символы"""
        if self.lowercase:
            text = text.lower()
        # оставляем только буквы и пробелы
        text = re.sub(r"[^а-яa-zё ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str):
        """Разбиваем текст на токены"""
        return text.split()

    def build_vocab(self, tokens):
        """Строим словарь по самым частотным словам"""
        counter = Counter(tokens)
        most_common = counter.most_common(self.vocab_size - 1)

        self.word2idx = {"<UNK>": 0}
        self.word2idx.update({word: i+1 for i, (word, _) in enumerate(most_common)})
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def encode(self, tokens):
        """Преобразуем токены в индексы"""
        return [self.word2idx.get(token, 0) for token in tokens]

    def decode(self, indices):
        """Преобразуем индексы обратно в токены"""
        return [self.idx2word.get(idx, "<UNK>") for idx in indices]

    def process_file(self, filepath: str):
        """Загрузка текста из файла и полная обработка"""
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        text = self.clean_text(text)
        tokens = self.tokenize(text)
        self.build_vocab(tokens)
        encoded = self.encode(tokens)

        return encoded

