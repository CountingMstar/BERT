# Dataset: 데이터의 샘플과 정답을 저장
from torch.utils.data import Dataset
import tqdm
import torch
import random

class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1
                
                if not on_memory:
                    self.file = open(corpus_path, "r", encoding=encoding)
                    self.random_file = open(corpus_path, "r", encoding=encoding)

                    for _ in range(random.randint(self.corpus_lines if self.corpus_lines))

