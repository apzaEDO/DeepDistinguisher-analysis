# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.model.transformer import init_linear_weights, Block, LayerNorm


class AngularEmbedding(nn.Module):
    def __init__(self, Q, dim, k=1):
        super().__init__()

        self.Q = Q
        self.proj_G = nn.Linear(2 * k, dim)
        self.proj_r = nn.Linear(2, dim)
        self.apply(init_linear_weights)

        self.k = k

    def to_xy(self, x, q=None):
        if not q:
            q = self.Q
        rad = x / q * 2 * torch.pi  # convert to radians
        return torch.cat((torch.cos(rad), torch.sin(rad)), dim=-1)

    def forward(self, G, r):
        # Embed A into x, y coordinates
        G = self.to_xy(G)
        r = self.to_xy(r)

        return torch.cat((self.proj_G(G), self.proj_r(r)), dim=1)


class CodeEmbedding(nn.Module):
    def __init__(self, k, input_dim, output_dim):
        super(CodeEmbedding, self).__init__()
        self.weight = nn.Parameter(
            torch.normal(mean=0.0, std=0.02, size=(k, input_dim, output_dim))
        )
        self.bias = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=(k, output_dim)))

    def forward(self, x):
        # x shape: [batch_size, k, input_dim]
        # Apply batch matrix multiplication:
        # weights shape: [k, input_dim, output_dim]
        # x shape after transpose: [batch_size, k, input_dim]
        # result shape: [batch_size, k, output_dim]

        result = torch.einsum("bkn,knd->bkd", x, self.weight)
        result += self.bias.unsqueeze(0)  # Add biases
        return result


class ProjEmbedding(nn.Module):
    def __init__(self, dim, G_dim):
        super().__init__()
        self.proj_G = nn.Linear(G_dim, dim)
        self.apply(init_linear_weights)

        self.G_dim = G_dim

    def forward(self, G):
        # Embed A into x, y coordinates
        # G shape B x k x mt -> B x k x d
        return self.proj_G(G)


class VEmbedding(nn.Module):
    def __init__(self, dim, G_dim, k):
        super().__init__()
        # Create a parameter tensor for k different projection matrices
        # Each matrix is of size G_dim x dim
        self.proj_G = nn.Parameter(torch.Tensor(k, G_dim, dim))
        self.k = k
        self.G_dim = G_dim
        self.dim = dim

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize the projection matrices
        nn.init.kaiming_uniform_(self.proj_G, a=math.sqrt(5))

    def forward(self, G):
        # G shape: B x k x G_dim
        B, k, G_dim = G.shape

        # Check if the input dimensions match
        assert (
            k == self.k and G_dim == self.G_dim
        ), "Input dimensions must match initialized dimensions"

        # Apply each projection matrix to corresponding slice of G
        # We use einsum for batch matrix multiplication: 'bki,kij->bkj'
        # b: batch size, k: number of matrices/slices, i: input dim, j: output dim
        result = torch.einsum("bki,kij->bkj", G, self.proj_G)

        return result


class BinaryHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, 1, bias=False)
        self.sig = nn.Sigmoid()
        self.apply(init_linear_weights)

    def forward(self, x):
        return self.proj(x)


class GDistTransformer(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.max_seq_len = 4096
        self.input_dim = params.model_input_dim
        self.input_len = params.model_input_len
        self.output_dim = params.model_output_dim
        self.output_len = params.model_output_len
        self.output_vocab_size = params.output_vocab_size

        if hasattr(params, "positional_encoding_nrows"):
            self.positional_encoding_nrows = params.positional_encoding_nrows
        else:
            self.positional_encoding_nrows = self.input_len
        self.positional_encoding_ncols = (
            self.input_len // self.positional_encoding_nrows
        )
        self._build_model(params)
        self._init_weights(params)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def _build_model(self, params):
        self.pos_emb_row = nn.Embedding(
            self.positional_encoding_nrows, params.enc_emb_dim
        )
        self.pos_emb_col = nn.Embedding(
            self.positional_encoding_ncols, params.enc_emb_dim
        )

        # self.tok_emb = CodeEmbedding(params.k, params.model_input_dim, params.enc_emb_dim)
        if hasattr(params, "tokenize") and params.tokenize:
            self.tok_emb = nn.Embedding(params.input_vocab_size, params.enc_emb_dim)
        else:
            self.tok_emb = ProjEmbedding(params.enc_emb_dim, params.model_input_dim)
        self.layers = nn.ModuleList([Block(params) for _ in range(params.n_enc_layers)])
        self.ln_f = LayerNorm(params.enc_emb_dim, bias=False)
        self.drop = nn.Dropout(params.dropout)

        self.head = nn.Linear(
            params.enc_emb_dim,
            params.model_output_dim * params.output_vocab_size,
            bias=False,
        )
        # BinaryHead(params.enc_emb_dim)

    def _init_weights(self, params):
        # init all weights
        self.apply(init_linear_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_enc_layers)
                )

    @property
    def num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def positional_embed(self, inputs):
        device = inputs.device
        _, k = inputs.shape[:2]
        pos = torch.arange(0, k, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)
        pos_row = pos // self.positional_encoding_ncols  # (1, t)
        pos_col = pos % self.positional_encoding_ncols
        pos_emb_row = self.pos_emb_row(pos_row)
        pos_emb_col = self.pos_emb_col(pos_col)
        return pos_emb_row + pos_emb_col

    def forward(self, inputs, labels=None, **kwargs):

        x = self.tok_emb(inputs) + self.positional_embed(inputs)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        pooled = torch.max(x, dim=1)[0]
        logits = self.head(pooled)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return dict(output=(logits > 0).to(int), loss=loss)


class GCompleteTransformer(GDistTransformer):
    def __init__(self, params):
        super().__init__(params)
        self.Q = params.Q
        self.loss_fn = torch.nn.CrossEntropyLoss(
            reduction="none"
        )  # BCEWithLogitsLoss(reduction='none')

    def _loss_fn(self, logits, labels, mask):
        loss = self.loss_fn(logits, labels)
        if mask is not None:
            loss = loss * mask.float()
            n_masked = mask.sum()
            loss = loss.sum() / n_masked
        else:
            loss = loss.mean()
        return loss

    def fwd(self, inputs):
        x = self.tok_emb(inputs) + self.positional_embed(inputs)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return x

    def forward(self, inputs, labels=None, masks=None):
        bs = len(inputs)
        x = self.fwd(inputs)

        logits = self.head(x)
        logits = logits.view(*labels.shape, self.output_vocab_size)
        logits = logits.permute(0, 3, 1, 2)  # (B, V, K, N)

        loss = None
        if labels is not None:
            loss = self._loss_fn(logits, labels.to(torch.long), masks)

        output = torch.argmax(logits, dim=1).to(torch.int)
        if masks is None:
            return dict(output=output, loss=loss)
        return dict(output=(output, masks), loss=loss)


class MatrixDistinguisher(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.max_seq_len = 4096
        self.input_dim = params.model_input_dim
        self.input_len = params.model_input_len
        self.output_dim = params.model_output_dim
        self.output_len = params.model_output_len
        self.output_vocab_size = params.output_vocab_size

        for arg in ["row_periods", "col_periods"]:
            if hasattr(params, arg) and getattr(params, arg):
                value = list(map(int, getattr(params, arg).split(";")))
                setattr(self, arg, value)
            else:
                setattr(self, arg, [])

        self._build_model(params)
        self._init_weights(params)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def _build_model(self, params):

        self.pos_emb_row_mod = []
        for period in self.row_periods:
            self.pos_emb_row_mod.append(nn.Embedding(period + 1, self.input_len))
        self.pos_emb_row_mod = nn.ModuleList(self.pos_emb_row_mod)

        self.pos_emb_row_div = []
        for period in self.row_periods:
            self.pos_emb_row_div.append(
                nn.Embedding(
                    (self.input_dim - self.input_len) // period + 1, self.input_len
                )
            )
        self.pos_emb_row_div = nn.ModuleList(self.pos_emb_row_div)

        self.pos_emb_col_mod = []
        for period in self.col_periods:
            self.pos_emb_col_mod.append(nn.Embedding(period, params.enc_emb_dim))
        self.pos_emb_col_mod = nn.ModuleList(self.pos_emb_col_mod)

        self.pos_emb_col_div = []
        for period in self.col_periods:
            self.pos_emb_col_div.append(
                nn.Embedding(self.input_len // period + 1, params.enc_emb_dim)
            )
        self.pos_emb_col_div = nn.ModuleList(self.pos_emb_col_div)

        # self.tok_emb = CodeEmbedding(params.k, params.model_input_dim, params.enc_emb_dim)
        if hasattr(params, "tokenize") and params.tokenize:
            self.tok_emb = nn.Embedding(params.input_vocab_size, params.enc_emb_dim)
        else:
            self.tok_emb = ProjEmbedding(params.enc_emb_dim, params.model_input_dim)
        self.layers = nn.ModuleList([Block(params) for _ in range(params.n_enc_layers)])
        self.ln_f = LayerNorm(params.enc_emb_dim, bias=False)
        self.drop = nn.Dropout(params.dropout)

        self.head = nn.Linear(
            params.enc_emb_dim,
            params.model_output_dim * params.output_vocab_size,
            bias=False,
        )

    def _init_weights(self, params):
        # init all weights
        self.apply(init_linear_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_enc_layers)
                )

    @property
    def num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def positional_embed(self, inputs):
        device = inputs.device
        _, k, n = inputs.shape
        col_pos = torch.arange(0, k, dtype=torch.long, device=device).unsqueeze(
            0
        )  # (1, k)
        col_emb = torch.tensor(0, dtype=torch.long, device=device)

        for i, period in enumerate(self.col_periods):
            col_pos_mod = col_pos % period  # (1, t)
            col_pos_div = col_pos // period
            pos_emb_col = self.pos_emb_col_mod[i](col_pos_mod) + self.pos_emb_col_div[
                i
            ](col_pos_div)
            col_emb = pos_emb_col + col_emb

        row_pos = torch.zeros(k, dtype=torch.long, device=device)

        row_emb = torch.tensor(0, dtype=torch.long, device=device)
        for i, period in enumerate(self.row_periods):
            row_pos_mod = torch.concatenate(
                (
                    period + row_pos,
                    torch.arange(0, n - k, dtype=torch.long, device=device) % period,
                )
            ).unsqueeze(0)

            row_pos_div = torch.concatenate(
                (
                    int(np.ceil((n - k) / period)) + row_pos,
                    torch.arange(0, n - k, dtype=torch.long, device=device) // period,
                )
            ).unsqueeze(0)

            pos_emb_row = self.pos_emb_row_mod[i](row_pos_mod) + self.pos_emb_row_div[
                i
            ](row_pos_div)
            row_emb = pos_emb_row + row_emb

        if len(self.row_periods):
            row_emb = row_emb.swapaxes(-2, -1)

        return row_emb, col_emb

    def forward(self, inputs, labels=None, key_padding_mask=None, **kwargs):
        # masque fourni ? sinon on le calcule
        valid = key_padding_mask
        if valid is None:
            valid = (inputs.abs().sum(dim=-1) != 0)  # (B,T)
    
        row_emb, seq_emb = self.positional_embed(inputs)
        x = self.tok_emb(inputs + row_emb) + seq_emb
        x = self.drop(x)
    
        for layer in self.layers:
            x = layer(x, key_padding_mask=valid)
    
        x = self.ln_f(x)
    
        x_masked = x.masked_fill(~valid[:, :, None], float("-inf"))
        pooled = x_masked.max(dim=1).values
    
        logits = self.head(pooled)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return dict(output=(logits > 0).to(int),logits=logits, loss=loss)

class MatrixDistinguisherA(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.max_seq_len = 4096
        self.input_dim = params.model_input_dim
        self.input_len = params.model_input_len
        self.output_dim = params.model_output_dim
        self.output_len = params.model_output_len
        self.output_vocab_size = params.output_vocab_size

        for arg in ["row_periods", "col_periods"]:
            if hasattr(params, arg) and getattr(params, arg):
                value = list(map(int, getattr(params, arg).split(";")))
                setattr(self, arg, value)
            else:
                setattr(self, arg, [])

        self._build_model(params)
        self._init_weights(params)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def _build_model(self, params):

        # Positional encoding des lignes de A
        # inputs.shape = (B, k, n-k)
        # params.model_input_len = k
        self.pos_emb = nn.Embedding(
            params.model_input_len,
            params.enc_emb_dim,
        )

        if hasattr(params, "tokenize") and params.tokenize:
            self.tok_emb = nn.Embedding(params.input_vocab_size, params.enc_emb_dim)
        else:
            self.tok_emb = ProjEmbedding(
                params.enc_emb_dim,
                params.model_input_dim,
            )

        self.layers = nn.ModuleList(
            [Block(params) for _ in range(params.n_enc_layers)]
        )

        self.ln_f = LayerNorm(params.enc_emb_dim, bias=False)
        self.drop = nn.Dropout(params.dropout)

        self.head = nn.Linear(
            params.enc_emb_dim,
            params.model_output_dim * params.output_vocab_size,
            bias=False,
        )
        
    def _init_weights(self, params):
        # init all weights
        self.apply(init_linear_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_enc_layers)
                )

    @property
    def num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def positional_embed(self, inputs):
        device = inputs.device
        _, k, _ = inputs.shape
        pos = torch.arange(0, k, dtype=torch.long, device=device).unsqueeze(0)
        return self.pos_emb(pos)

    def forward(self, inputs, labels=None, key_padding_mask=None, **kwargs):
        # masque fourni ? sinon on le calcule
        valid = key_padding_mask
        if valid is None:
            valid = (inputs.abs().sum(dim=-1) != 0)  # (B,T)
    
        seq_emb = self.positional_embed(inputs)
        x = self.tok_emb(inputs.float()) + seq_emb
        x = self.drop(x)
    
        for layer in self.layers:
            x = layer(x, key_padding_mask=valid)
    
        x = self.ln_f(x)
    
        x_masked = x.masked_fill(~valid[:, :, None], float("-inf"))
        pooled = x_masked.max(dim=1).values
    
        logits = self.head(pooled)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return dict(output=(logits > 0).to(int),logits=logits, loss=loss)

class MatrixDistinguisherH(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.max_seq_len = 4096
        self.input_dim = params.model_input_dim
        self.input_len = params.model_input_len
        self.output_dim = params.model_output_dim
        self.output_len = params.model_output_len
        self.output_vocab_size = params.output_vocab_size

        for arg in ["row_periods", "col_periods"]:
            if hasattr(params, arg) and getattr(params, arg):
                value = list(map(int, getattr(params, arg).split(";")))
                setattr(self, arg, value)
            else:
                setattr(self, arg, [])

        self._build_model(params)
        self._init_weights(params)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def _build_model(self, params):

        self.pos_emb_row_mod = []
        for period in self.row_periods:
            self.pos_emb_row_mod.append(nn.Embedding(period + 1, self.input_len))
        self.pos_emb_row_mod = nn.ModuleList(self.pos_emb_row_mod)

        self.pos_emb_row_div = []
        for period in self.row_periods:
            self.pos_emb_row_div.append(
                nn.Embedding(
                    (self.input_dim - self.input_len) // period + 1, self.input_len
                )
            )
        self.pos_emb_row_div = nn.ModuleList(self.pos_emb_row_div)

        self.pos_emb_col_mod = []
        for period in self.col_periods:
            self.pos_emb_col_mod.append(nn.Embedding(period, params.enc_emb_dim))
        self.pos_emb_col_mod = nn.ModuleList(self.pos_emb_col_mod)

        self.pos_emb_col_div = []
        for period in self.col_periods:
            self.pos_emb_col_div.append(
                nn.Embedding(self.input_len // period + 1, params.enc_emb_dim)
            )
        self.pos_emb_col_div = nn.ModuleList(self.pos_emb_col_div)

        # self.tok_emb = CodeEmbedding(params.k, params.model_input_dim, params.enc_emb_dim)
        if hasattr(params, "tokenize") and params.tokenize:
            self.tok_emb = nn.Embedding(params.input_vocab_size, params.enc_emb_dim)
        else:
            self.tok_emb = ProjEmbedding(params.enc_emb_dim, params.model_input_dim)
        self.layers = nn.ModuleList([Block(params) for _ in range(params.n_enc_layers)])
        self.ln_f = LayerNorm(params.enc_emb_dim, bias=False)
        self.drop = nn.Dropout(params.dropout)

        self.head = nn.Linear(
            params.enc_emb_dim,
            params.model_output_dim * params.output_vocab_size,
            bias=False,
        )

    def _init_weights(self, params):
        # init all weights
        self.apply(init_linear_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_enc_layers)
                )

    @property
    def num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def positional_embed(self, inputs):
        device = inputs.device
        _, k, n = inputs.shape
        col_pos = torch.arange(0, k, dtype=torch.long, device=device).unsqueeze(
            0
        )  # (1, k)
        col_emb = torch.tensor(0, dtype=torch.long, device=device)

        for i, period in enumerate(self.col_periods):
            col_pos_mod = col_pos % period  # (1, t)
            col_pos_div = col_pos // period
            pos_emb_col = self.pos_emb_col_mod[i](col_pos_mod) + self.pos_emb_col_div[
                i
            ](col_pos_div)
            col_emb = pos_emb_col + col_emb

        row_pos = torch.zeros(k, dtype=torch.long, device=device)

        row_emb = torch.tensor(0, dtype=torch.long, device=device)
        for i, period in enumerate(self.row_periods):
            row_pos_mod = torch.concatenate(
                (
                    period + row_pos,
                    torch.arange(0, n - k, dtype=torch.long, device=device) % period,
                )
            ).unsqueeze(0)

            row_pos_div = torch.concatenate(
                (
                    int(np.ceil((n - k) / period)) + row_pos,
                    torch.arange(0, n - k, dtype=torch.long, device=device) // period,
                )
            ).unsqueeze(0)

            pos_emb_row = self.pos_emb_row_mod[i](row_pos_mod) + self.pos_emb_row_div[
                i
            ](row_pos_div)
            row_emb = pos_emb_row + row_emb

        if len(self.row_periods):
            row_emb = row_emb.swapaxes(-2, -1)

        return row_emb, col_emb

    def forward(self, inputs, labels=None, **kwargs):
        row_emb, seq_emb = self.positional_embed(inputs)
        x = self.tok_emb(inputs + row_emb) + seq_emb
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        pooled = torch.max(x, dim=1)[0]
        logits = self.head(pooled)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return dict(output=(logits > 0).to(int), loss=loss)


class MatrixViewEncoder(nn.Module):
    """
    Encode une vue matricielle :
      - vue G : G = [I_k | A]   -> identity_side="left"
      - vue H : H = [A^T | I_r] -> identity_side="right"
    où r = n-k.
    Chaque token = une ligne entière de la matrice.
    """

    def __init__(self, params, n_rows, n_cols, identity_size, identity_side="left"):
        super().__init__()
        assert identity_side in {"left", "right"}

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.identity_size = identity_size
        self.identity_side = identity_side
        self.enc_emb_dim = params.enc_emb_dim

        for arg in ["row_periods", "col_periods"]:
            if hasattr(params, arg) and getattr(params, arg):
                value = list(map(int, getattr(params, arg).split(";")))
                setattr(self, arg, value)
            else:
                setattr(self, arg, [])

        self._build_model(params)
        self._init_weights(params)

    def _build_model(self, params):
        nonsys_cols = self.n_cols - self.identity_size

        # Embeddings structurels sur les colonnes de la ligne (avant projection)
        self.pos_emb_row_mod = nn.ModuleList([
            nn.Embedding(period + 1, self.n_rows)
            for period in self.row_periods
        ])

        self.pos_emb_row_div = nn.ModuleList([
            nn.Embedding((nonsys_cols + period - 1) // period + 1, self.n_rows)
            for period in self.row_periods
        ])

        # Embeddings de position sur l'index de ligne (séquence de tokens)
        self.pos_emb_col_mod = nn.ModuleList([
            nn.Embedding(period, params.enc_emb_dim)
            for period in self.col_periods
        ])

        self.pos_emb_col_div = nn.ModuleList([
            nn.Embedding(self.n_rows // period + 1, params.enc_emb_dim)
            for period in self.col_periods
        ])

        # Même logique que ton code d'origine
        if hasattr(params, "tokenize") and params.tokenize:
            self.tok_emb = nn.Embedding(params.input_vocab_size, params.enc_emb_dim)
        else:
            self.tok_emb = ProjEmbedding(params.enc_emb_dim, self.n_cols)

        self.layers = nn.ModuleList([Block(params) for _ in range(params.n_enc_layers)])
        self.ln_f = LayerNorm(params.enc_emb_dim, bias=False)
        self.drop = nn.Dropout(params.dropout)

    def _init_weights(self, params):
        self.apply(init_linear_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_enc_layers)
                )

    def positional_embed(self, inputs):
        """
        inputs: (B, r, n)
        row_emb: (1, r, n)      -> ajouté aux valeurs de la ligne avant projection
        seq_emb: (1, r, d)      -> position de la ligne dans la séquence
        """
        device = inputs.device
        B, r, n = inputs.shape

        assert r == self.n_rows, f"expected {self.n_rows} rows, got {r}"
        assert n == self.n_cols, f"expected {self.n_cols} cols, got {n}"

        # Position des lignes (séquence de tokens)
        col_pos = torch.arange(0, r, dtype=torch.long, device=device).unsqueeze(0)  # (1, r)
        seq_emb = torch.zeros(1, r, self.enc_emb_dim, device=device)

        for i, period in enumerate(self.col_periods):
            col_pos_mod = col_pos % period
            col_pos_div = col_pos // period
            seq_emb = seq_emb + self.pos_emb_col_mod[i](col_pos_mod) + self.pos_emb_col_div[i](col_pos_div)

        # Embedding structurel sur les colonnes de chaque ligne
        row_emb = torch.zeros(1, r, n, device=device, dtype=torch.float32)

        nonsys_cols = n - self.identity_size
        special_div_value = (nonsys_cols + self.row_periods[0] - 1) // self.row_periods[0] if len(self.row_periods) else 0

        for i, period in enumerate(self.row_periods):
            nonsys_idx = torch.arange(0, nonsys_cols, dtype=torch.long, device=device)
            nonsys_mod = nonsys_idx % period
            nonsys_div = nonsys_idx // period

            special_mod = torch.full((self.identity_size,), period, dtype=torch.long, device=device)
            special_div = torch.full(
                (self.identity_size,),
                (nonsys_cols + period - 1) // period,
                dtype=torch.long,
                device=device,
            )

            if self.identity_side == "left":
                row_pos_mod = torch.cat([special_mod, nonsys_mod], dim=0).unsqueeze(0)  # (1, n)
                row_pos_div = torch.cat([special_div, nonsys_div], dim=0).unsqueeze(0)
            else:
                row_pos_mod = torch.cat([nonsys_mod, special_mod], dim=0).unsqueeze(0)
                row_pos_div = torch.cat([nonsys_div, special_div], dim=0).unsqueeze(0)

            # -> (1, n, r), puis transpose en (1, r, n)
            pos_emb_row = self.pos_emb_row_mod[i](row_pos_mod) + self.pos_emb_row_div[i](row_pos_div)
            row_emb = row_emb + pos_emb_row.swapaxes(-2, -1)

        return row_emb, seq_emb

    def forward(self, inputs):
        row_emb, seq_emb = self.positional_embed(inputs)

        if isinstance(self.tok_emb, nn.Embedding):
            # Dans ta logique actuelle, tokenize=True n'est pas vraiment compatible
            # avec l'ajout d'un row_emb flottant avant embedding.
            if len(self.row_periods) > 0:
                raise ValueError("tokenize=True est incompatible ici avec row_periods non vides.")
            x = self.tok_emb(inputs.long())
        else:
            x = self.tok_emb(inputs.float() + row_emb)

        x = x + seq_emb
        x = self.drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)

        # pooling comme dans ton code
        pooled = torch.max(x, dim=1)[0]   # (B, d)
        return pooled


class MatrixDistinguisherGH(nn.Module):
    """
    Reçoit G et H en entrée.
    Hypothèse typique :
      - G shape = (B, k, n), avec G = [I_k | A]
      - H shape = (B, n-k, n), avec H = [A^T | I_{n-k}]
    """

    def __init__(self, params):
        super().__init__()

        n = params.model_input_dim
        k = params.model_input_len
        r = n - k
        d = params.enc_emb_dim

        self.max_seq_len = 4096
        self.input_dim = n
        self.input_len = k
        self.output_dim = params.model_output_dim
        self.output_len = params.model_output_len
        self.output_vocab_size = params.output_vocab_size

        # Branche G : bloc identité à gauche
        self.g_encoder = MatrixViewEncoder(
            params=params,
            n_rows=k,
            n_cols=n,
            identity_size=k,
            identity_side="left",
        )

        # Branche H : bloc identité à droite
        self.h_encoder = MatrixViewEncoder(
            params=params,
            n_rows=r,
            n_cols=n,
            identity_size=r,
            identity_side="right",
        )

        # Fusion tardive avec interactions explicites
        self.fusion = nn.Sequential(
            nn.Linear(4 * d, 2 * d, bias=False),
            nn.GELU(),
            nn.Dropout(params.dropout),
            nn.Linear(2 * d, params.model_output_dim * params.output_vocab_size, bias=False),
        )

        self.fusion.apply(init_linear_weights)
        self.loss_fn = nn.BCEWithLogitsLoss()

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, inputs_G=None, inputs_H=None, labels=None, **kwargs):
        """
        Compatible avec un dataloader qui renvoie par exemple :
            {
                "inputs_G": G,
                "inputs_H": H,
                "labels": y,
            }
        """
        if inputs_G is None:
            inputs_G = kwargs.get("inputs_G", None)
        if inputs_H is None:
            inputs_H = kwargs.get("inputs_H", None)

        if inputs_G is None or inputs_H is None:
            raise ValueError("Il faut fournir inputs_G et inputs_H.")

        z_g = self.g_encoder(inputs_G)   # (B, d)
        z_h = self.h_encoder(inputs_H)   # (B, d)

        # Fusion simple mais efficace
        z = torch.cat(
            [
                z_g,
                z_h,
                z_g * z_h,
                torch.abs(z_g - z_h),
            ],
            dim=-1,
        )

        logits = self.fusion(z)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return dict(output=(logits > 0).to(int), loss=loss)


    def create(cls, params, tokenizer):
        assert params.eval_samples > 0

        dataset_size = params.eval_samples + params.train_samples
        dataset = cls(tokenizer, params, dataset_size)

        train, test = len(dataset) - params.eval_samples, params.eval_samples

        train_dataset, test_dataset = random_split(
            dataset, [train, test], generator=torch.Generator().manual_seed(42)
        )
        return train_dataset, test_dataset