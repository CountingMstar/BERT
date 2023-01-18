import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(query.size(-1))

        if mask is not None:
            # masked_fill: mask가 0인 부분을 -1e9로 채움
            scores = scores.masked_fill(mask == 0, -1e9)
        
        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            # ex) dropout = nn.Dropout(0.25): 25% 드롭아웃
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

        