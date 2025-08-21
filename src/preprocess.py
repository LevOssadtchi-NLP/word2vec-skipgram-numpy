import re
from collections import Counter

class Preprocessor:
    def __init__(self, vocab_size=10000, lowercase=True):
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.word2idx = {}
        self.idx2word = {}

    def clean_text(self, text: str) -> str:
        """Простая очистка текста"""
        if self.lowercase:
            text = text.lower()
        text = re.sub(r"[^а-яa-zё ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str):
        """Разделяем текст на токены по пробелам"""
        return text.split()

    def build_vocab(self, tokens):
        """Строим словарь по самым частотным словам:
        - резервируем индекс 0 для <UNK>
        - нумеруем остальные слова
        """
        counter = Counter(tokens)
        most_common = counter.most_common(self.vocab_size - 1)
        
        self.word2idx = {"<UNK>": 0}
        self.word2idx.update({word: i+1 for i, (word, _) in enumerate(most_common)})
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        
        print(self.word2idx)
        print(self.idx2word)

    def encode(self, tokens):
        """Преобразуем список слов в список индексов"""
        return [self.word2idx.get(token, 0) for token in tokens]

    def decode(self, indices):
        """Преобразуем список индексов обратно в слова"""
        return [self.idx2word.get(idx, "<UNK>") for idx in indices]

    def process_file(self, filepath: str):
        """Полная обработка файла:
        - читаем текст
        - чистим и токенизируем
        - строим словарь
        - возвращаем текст в виде индексов
        """
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        text = self.clean_text(text)
        tokens = self.tokenize(text)
        self.build_vocab(tokens)
        encoded = self.encode(tokens)

        return encoded
