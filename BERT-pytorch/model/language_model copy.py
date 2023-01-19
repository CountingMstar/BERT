import torch.nn as nn
from .bert import BERT

class BERTLM(nn.Module):
    def __init__(self, bert: BERT, vocab_size):
        # super().__init__(): 다른 클래스의 속성 및 메소드를 자동으로 불러와 해당 클래스에서 사용이 가능하도록 함
        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)

        return self.next_sentence(x), self.mask_lm(x)


class NextSentencePrediction(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        # NSP는 이진분류 -> 2
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    def __init__(self, hidden, vocab_size):
        super().__init__()
        # MLM은 n-gram 분류 -> n=vocab_size
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x