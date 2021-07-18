#!/usr/bin/python
#-*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        #self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        #attn = self.dropout(attn)
        outputs = torch.bmm(attn, v)

        return outputs

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, num_heads, q_dim, k_dim, d_in, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.q_dim = q_dim
        self.k_dim = k_dim
        
        self.d_in = d_in

        self.w_qs = nn.Linear(q_dim, num_heads * d_in)
        self.w_ks = nn.Linear(k_dim, num_heads * d_in)
        self.w_vs = nn.Linear(k_dim, num_heads * d_in)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (q_dim + d_in)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (k_dim + d_in)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (k_dim + d_in)))

        self.attention = ScaledDotProductAttention(temperature=np.power(k_dim, 0.5))
        self.layer_norm = nn.LayerNorm(q_dim)

        self.fc = nn.Linear(num_heads * d_in, q_dim)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # k and v should be the same
        d_in, num_heads = self.d_in, self.num_heads

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, num_heads, d_in)
        k = self.w_ks(k).view(sz_b, len_k, num_heads, d_in)
        v = self.w_vs(v).view(sz_b, len_v, num_heads, d_in)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_in) # (n*b) x lq x dq
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_in) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_in) # (n*b) x lv x dk

        mask = mask.repeat(num_heads, 1, 1) # (n*b) x .. x ..
        #outputs, attn = self.attention(q, k, v, mask=mask)
        outputs = self.attention(q, k, v, mask=mask)

        outputs = outputs.view(num_heads, sz_b, len_q, d_in)
        outputs = outputs.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dk)

        outputs = self.dropout(self.fc(outputs))
        outputs = self.layer_norm(outputs + residual)

        return outputs

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        outputs = x.transpose(1, 2)
        outputs = self.w_2(F.relu(self.w_1(outputs)))
        outputs = outputs.transpose(1, 2)
        outputs = self.dropout(outputs)
        outputs = self.layer_norm(outputs + residual)
        return outputs


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        #return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)
        return position * np.exp( (-np.log(10000.0)) * 2.0 * (hid_idx//2) / d_hid)
    
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    assert seq_k.dim() == 3

    len_q = seq_q.size(1)
    padding_mask = torch.sum(torch.abs(seq_k), -1).eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_non_pad_mask(seq):
    assert seq.dim() == 3
    return torch.sum(torch.abs(seq), -1).ne(0).type(torch.float).unsqueeze(-1)

