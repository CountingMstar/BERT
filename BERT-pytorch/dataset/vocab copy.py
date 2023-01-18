import pickle
import tqdm
from collections import Counter

class TorchVocab(object):
    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'], 
                vectors=None, unk_init=None, vectors_cache=None):
        # counter: 글자수 세기
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)

        # special token을 counter에서 지워 카운트하지 않기
        for tok in specials:
            del counter[tok]
        
        max_size = None if max_size is None else max_size + len(self.itos)

        # counter의 단어들을 빈도수에 따라 정렬
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)
        
        # list self.itos를 딕셔너리형태로 변환
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            # ???
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            # assert: 뒤의 조건문이 True가 아니면 AssertError발생
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        
        # 상속받은 부모 클래스
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"],
                        max_size=max_size, min_freq=min_freq)

        def to_seq(self, sentence, seq_len, with_eos=False, with_sos=False) -> list:
            pass

        def from_seq(self, seq, join=False, with_pad=False):
            pass
        
        # vocab 불러오기 또는 저장
        @staticmethod
        def load_vocab(vocab_path: str) -> 'Vocab':
            with open(vocab_path, "rb") as f:
                return pickle.load(f)

        def save_vocab(self, vocab_path):
            with open(vocab_path, "wb") as f:
                pickle.dump(self, f)


class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        print("Building Vocab")
        counter = Counter()

        for line in tqdm.tqdm(texts):
            # isinstance(a, list): a가 list형인지 판별
            if isinstance(line, list):
                words = line
            else:
                # 교체작업
                words = line.replace("\n", "").replace("\t", "").split()

            for word in words:
                counter[word] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()
        
        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.eos_index]
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]
        
        return (seq, origin_seq_len if with_len else seq)

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx]
                if idx < len(self.itos)
                else "<%d>" % idx
                for idx in seq
                if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)


def build():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus_path", require=True, type=str)
    parser.add_argument("-o", "--output_path", required=True, type=str)
    parser.add_argument("-s", "--vocab_size", type=int, default=None)
    parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    parser.add_argument("-m", "--min_freq", type=int, default=1)
    args = parser.parse_args()

    with open(args.corpus_path, "r", encoding=args.encoding) as f:
        vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq)

    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(args.output_path)
