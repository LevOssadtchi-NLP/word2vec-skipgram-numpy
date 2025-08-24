import os
import numpy as np
from preprocess import Preprocessor
from dataset import SkipGramDataset
from model import SkipGram
from utils import save_embeddings

def main():
    # Параметры
    data_path = "../data/ruwiki_sample.txt"
    embedding_dim = 50
    window_size = 2
    epochs = 30
    neg_sampling = True   # включить/выключить Negative Sampling
    neg_samples = 5

    # Загрузка корпуса
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Препроцессинг
    preprocessor = Preprocessor(vocab_size=10000)
    text = preprocessor.clean_text(text)
    tokenized = preprocessor.tokenize(text)[:500000]
    preprocessor.build_vocab(tokenized)
    encoded = preprocessor.encode(tokenized)

    print(f"Размер словаря: {len(preprocessor.word2idx)}")

    # Датасет
    dataset = SkipGramDataset(encoded, window_size)

    # Модель
    model = SkipGram(
        vocab_size=len(preprocessor.word2idx),
        embedding_dim=embedding_dim,
        learning_rate=0.01,
        neg_sampling=neg_sampling,
        neg_samples=neg_samples
    )

    # Обучение
    model.train(dataset, epochs=epochs)

    # Сохранение эмбеддингов
    os.makedirs("artifacts", exist_ok=True)
    save_embeddings(model.W, preprocessor.idx2word, "artifacts/embeddings.csv")

    print("Обучение завершено. Эмбеддинги сохранены в artifacts/embeddings.csv")


if __name__ == "__main__":
    main()

