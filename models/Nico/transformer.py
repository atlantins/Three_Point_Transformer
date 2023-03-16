import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# reference https://github.com/jadore801120/attention-is-all-you-need-pytorch
# 非常重要的一点，q,k的隐藏维度要相同，k,v的数量要相同

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, q, k, v):
        # 这一步就说明了，只能通过-1,-2分别得为维度和数量，这也是为什么要把n_heads变到前面去的原因
        attn = q @ k.transpose(-1, -2)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-1)
        output = attn @ v

        return output, attn
        
        
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model_q, d_model_kv, d_k, d_v):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        
        self.w_qs = nn.Linear(d_model_q, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model_kv, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model_kv, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model_q, bias=False)

        self.attention = Attention()

        self.layer_norm1 = nn.LayerNorm(n_head * d_v, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model_q, eps=1e-6)


    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        b_size, n_q, n_k = q.size(0), q.size(1), k.size(1)

        residual = q

        # Pass through the pre-attention projection: b x k x (n*dv)
        # Separate different heads: b x k x n x dv
        
        # 因为n被占用了，所以点云的数量用k表示
        # q,k 进去之前是 b,k,d_q -> b,k,n*d_k -> b,k,n,d_k ->
        # v                     -> b,k,n*d_v -> b,k,n,d_v
        q = self.w_qs(q).view(-1, n_q, n_head, d_k)
        k = self.w_ks(k).view(-1, n_k, n_head, d_k)
        v = self.w_vs(v).view(-1, n_k, n_head, d_v)
        
        # b,k,n,d_k -> b,n,k,d_k
        # b,k,n,d_v -> b,n,k,d_v
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # get b x n x k x dv
        q, attn = self.attention(q, k, v)
        
        # b x k x ndv
        q = q.transpose(1, 2).contiguous().view(b_size, n_q, -1)
        s = self.layer_norm1(residual + q)
        res = self.layer_norm2(s + self.fc(s))

        return res, attn