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

                    for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                        self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        # 랜덤으로 뽑은 문장
        t1, t2, is_next_label = self.random_sent(item)
        
        # 위 문장의 단어 인덱스
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # 문장의 앞, 뒤에 시작, 끝 토큰 붙이기
        t1 = [self.vocab.sos_index] + t1_label + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        # input의 세그먼트(문장1, 문장2)
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        # bert input
        bert_input = (t1 + t2)[:self.seq_len]
        # ???
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            # random.random(): 0 ~ 1사이의 난수 생성
            prob = random.random()
            # 15%만 변환
            if prob < 0.15:
                prob /= 0.15
                # 그중 80%는 마스킹
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index
                # 10%는 다른 임의의 단어로 변환
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))
                # 나머지 10%는 unkown 처리(원래 논문에서는 원래 그대로)
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            # 나머지 85%는 불변환 
            # ???
            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)
        
        return tokens, output_label

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)
        
        # 50%는 isnext, 나머지 50%는 notnext
        # t1, t2는 이어진 문장을 나눈것
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]
        
        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()        
        return line[:-1].split("\t")[1]

        