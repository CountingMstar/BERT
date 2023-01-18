import torch.nn as nn
from .single import Attention

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        # 상속 인스턴스
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        # nn.Sequential: module끼리 순차적으로 연결
        # nn.ModuleList: module끼리 연결안됨
        # query, key, value 이렇게 3개
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # view: tensor차원 변환
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                            for l, x in zip(self.linear_layers, (query, key, value))]
        
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # contiguous: 비연속적인 텐서를 연속적 텐서로 변환
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
