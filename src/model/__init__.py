# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from src.model.distinguisher_transformer import (
    GCompleteTransformer,
    MatrixDistinguisher,
    MatrixDistinguisherH,
    MatrixDistinguisherGH,
    MatrixDistinguisherA
)
from src.model.models import ConvNet, LRNNet, LRNet, NNet, AttNet
from src.utils import model_size
from src.model.transformer import BaseEncoder, EncoderParams


logger = getLogger()


def get_model(params):
    if params.task.startswith("code-dist-gh"):
        return MatrixDistinguisherGH(params)
        
    elif "symbolic" in params.task:
        enc_params = EncoderParams(
            enc_emb_dim=params.enc_emb_dim,
            n_enc_layers=params.n_enc_layers,
            n_enc_heads=params.n_enc_heads,
            input_vocab_size=params.input_vocab_size,
            output_vocab_size=params.output_vocab_size,
            output_len=params.model_output_len,
            dropout=params.dropout,
            attention_dropout=params.attention_dropout,
        )
        model = BaseEncoder(enc_params)
    elif params.task.startswith("code-complete"):
        model = GCompleteTransformer(params)

    elif params.task.startswith("code") or params.task.startswith("view"):
        if params.model == "encoder":
            if params.representation in ["H","T","HT"]:
                model = MatrixDistinguisherH(params)
            elif params.representation in ["G","GT","H","HT"]:
                model = MatrixDistinguisher(params)
            elif params.representation in ["A","AT"]:
                model=MatrixDistinguisherA(params)
            elif params.representation =="GH" :
                model = MatrixDistinguisherGH(params)
            else :
                print("Erreur du choix de représentation")
        elif params.model == "lr":
            model = LRNet(params)
        elif params.model == "nnet":
            # model = ConvNet(params)
            model = NNet(params)
        elif params.model == "cnet":
            model = ConvNet(params)
        elif params.model == "lrnet":
            model = LRNNet(params)
        elif params.model == "attnet":
            model = AttNet(params)
        else:
            raise NotImplementedError()

    else:
        raise NotImplementedError(
            f"Model {params.model} for task {params.task} not implemented"
        )

    params.model_size = model_size(model)
    logger.info(f"Model Size: {params.model_size}")

    return model
