import torch.nn as nn
import torch
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()
        # require_grad: 역전파중 계산을 하지 않겠다. 즉 파라미터를 업데이트 하지 않겠다.
        pe.require_grad = False 

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register_buffer: optimizer가 업데이트하지 않는다. 그러나 연산은 가능 
        # 모델 매개 변수로 간주되지 않는 버퍼를 등록하는데 사용
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # ???

    def forward(self, x):
        return self.pe[:, :x.size(1)]
