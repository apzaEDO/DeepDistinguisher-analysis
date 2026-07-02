import math
import torch
import torch.nn as nn
import numpy as np

from code_article.src.model.transformer import init_linear_weights, Block, LayerNorm
from code_article.src.model.distinguisher_transformer import ProjEmbedding


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
            nn.Embedding(nonsys_cols // period + 1, self.n_rows)
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