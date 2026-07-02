"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Full definition of an encoder-only transformer.
Inspired by Andrej Karpathy's nanoGPT, originally licensed under MIT.
https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

from abc import abstractmethod
from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange


@dataclass
class EncoderParams:
    enc_emb_dim: int
    n_enc_layers: int
    n_enc_heads: int
    input_vocab_size: int
    output_vocab_size: int
    output_len: int
    dropout: float
    attention_dropout: float
    max_seq_len: int = 4096


class BaseEncoder(nn.Module):
    def __init__(self, params: EncoderParams):
        super().__init__()
        self.params = params
        self.max_seq_len = params.max_seq_len

        self.tok_emb = self.get_input_embedding(params)
        self.head = self.get_head(params)

        self.pos_emb = nn.Embedding(params.max_seq_len, params.enc_emb_dim)
        self.drop = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList([Block(params) for _ in range(params.n_enc_layers)])
        self.ln_f = LayerNorm(params.enc_emb_dim, bias=False)

        self.loss_fn = (
            torch.nn.CrossEntropyLoss()
            if params.output_vocab_size > 2
            else torch.nn.BCEWithLogitsLoss()
        )

        self._init_parameters(params.n_enc_layers)

    def get_input_embedding(self, params):
        return nn.Embedding(params.input_vocab_size, params.enc_emb_dim)

    def get_head(self, params):
        return DigitHead(
            params.output_len, params.enc_emb_dim, params.output_vocab_size
        )

    def forward(self, inputs, labels=None, **kwargs):
        device = inputs.device
        t = inputs.shape[1]

        err_msg = f"seq len {t} too long, max seq len is {self.max_seq_len}"
        assert t <= self.max_seq_len, err_msg

        tok_emb = self.tok_emb(inputs)  # (b, t, enc_emb_dim)

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)
        pos_emb = self.pos_emb(pos)  # (1, t, enc_emb_dim)

        x = self.drop(tok_emb + pos_emb)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        pooled = torch.max(x, dim=1)[0]
        logits = self.head(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
        # output if BCE ( binary classification) :
        if self.params.output_vocab_size > 2:
            output = logits.max(dim=1)[1]
        else:
            output = logits > 0
        return dict(output=output, loss=loss)

    def _init_parameters(self, nlayers):
        # init all weights
        self.apply(init_linear_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * nlayers))


def init_linear_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def new_gelu(x):
    """
    Implementation of the GELU activation function in Google BERT (same as OpenAI GPT).
    Ref: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttentionInitial(nn.Module):
    def __init__(self, params):
        super().__init__()
        assert params.enc_emb_dim % params.n_enc_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(params.enc_emb_dim, 3 * params.enc_emb_dim, bias=False)
        # output projection
        self.c_proj = nn.Linear(params.enc_emb_dim, params.enc_emb_dim, bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(params.attention_dropout)
        self.resid_dropout = nn.Dropout(params.dropout)
        self.n_enc_heads = params.n_enc_heads
        self.enc_emb_dim = params.enc_emb_dim
        self.dropout = params.dropout

    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (enc_emb_dim)
        B, T, C = x.size()

        # Calculate query, key, values for all heads in batch
        # Move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.enc_emb_dim, dim=2)
        # (B, nh, T, hs)
        k = k.view(B, T, self.n_enc_heads, C // self.n_enc_heads).transpose(1, 2)
        q = q.view(B, T, self.n_enc_heads, C // self.n_enc_heads).transpose(1, 2)
        v = v.view(B, T, self.n_enc_heads, C // self.n_enc_heads).transpose(1, 2)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class SelfAttention(nn.Module):
    def __init__(self, params):
        super().__init__()
        assert params.enc_emb_dim % params.n_enc_heads == 0
        self.c_attn = nn.Linear(params.enc_emb_dim, 3 * params.enc_emb_dim, bias=False)
        self.c_proj = nn.Linear(params.enc_emb_dim, params.enc_emb_dim, bias=False)
        self.attn_dropout = nn.Dropout(params.attention_dropout)
        self.resid_dropout = nn.Dropout(params.dropout)
        self.n_enc_heads = params.n_enc_heads
        self.enc_emb_dim = params.enc_emb_dim

    def forward(self, x, key_padding_mask=None):
        """
        x: (B,T,C)
        key_padding_mask: (B,T) bool, True = token valide, False = padding
        """
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.enc_emb_dim, dim=2)
        k = k.view(B, T, self.n_enc_heads, C // self.n_enc_heads).transpose(1, 2)  # (B,H,T,hs)
        q = q.view(B, T, self.n_enc_heads, C // self.n_enc_heads).transpose(1, 2)
        v = v.view(B, T, self.n_enc_heads, C // self.n_enc_heads).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B,H,T,T)

        if key_padding_mask is not None:
            # masque sur les KEYS (dernière dim): on interdit d'attendre le padding
            scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))

        att = F.softmax(scores, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, dim, factor=4, dropout=0.0):
        super().__init__()
        inner_dim = int(factor * dim)
        self.c_fc = nn.Linear(dim, inner_dim, bias=False)
        self.c_proj = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class BlockInitial(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.ln_1 = LayerNorm(params.enc_emb_dim, bias=False)
        self.attn = SelfAttention(params)
        self.ln_2 = LayerNorm(params.enc_emb_dim, bias=False)
        self.mlp = MLP(params.enc_emb_dim, dropout=params.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Block(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.ln_1 = LayerNorm(params.enc_emb_dim, bias=False)
        self.attn = SelfAttention(params)
        self.ln_2 = LayerNorm(params.enc_emb_dim, bias=False)
        self.mlp = MLP(params.enc_emb_dim, dropout=params.dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.ln_1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class DigitHead(nn.Module):
    """Predicts n digits from a representation using a single linear layer."""

    def __init__(self, digits, in_dim, n_symbols):
        super().__init__()
        # Single linear layer with combined output dimensions
        self.head = nn.Linear(in_dim, digits * n_symbols, bias=False)
        self.digits = digits
        self.n_symbols = n_symbols

    def forward(self, x):
        output = self.head(x)
        # Reshape the output to separate the digits and symbols
        # Assuming x.shape is (batch_size, in_dim)
        logits = (
            output.view(-1, self.n_symbols, self.digits)
            if self.digits > 1
            else output.view(-1, self.n_symbols)
        )
        return logits
