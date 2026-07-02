# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from typing import Union
import torch
import numpy as np

logger = logging.getLogger("tokenizer")
SPECIAL_WORDS = ["<eos>", "<pad>", "<mask>", "x", "/", "+", "-"]


class BaseCodeTokenizer:
    data_dim_multiplier: int = 1
    data_len_multiplier: int = 1

    def __call__(self, inputs: Union[list, np.ndarray], labels=False) -> torch.Tensor:
        if not isinstance(inputs, np.ndarray):
            inputs = np.stack(inputs)

        inputs = torch.from_numpy(inputs)
        inputs = inputs.to(torch.float)
        return inputs


class FqTokenizer(BaseCodeTokenizer):
    def __init__(self, params):
        self.Q = params.Q

        # assume q prime
        self.is_twopower = False

        # else only powers of 2 are supported
        self.logq = int(np.log2(self.Q))
        if self.logq == np.log2(self.Q) and self.logq > 1:
            self.is_twopower = True
            logger.info(f"Q {self.Q} is a power of 2, embedding into vectors")

        if self.is_twopower:
            self.data_len_multiplier *= self.logq
        elif self.Q > 2:
            self.data_dim_multiplier *= 2  # angular embedding

    def __call__(self, inputs, labels=False):
        if labels:
            return super().__call__(inputs, labels)

        inputs = np.stack(inputs)

        if self.is_twopower and self.Q > 2:
            bs, n, d = inputs.shape
            binary_matrix = np.unpackbits(
                np.expand_dims(inputs, axis=2).astype(np.uint8),
                axis=2,
                bitorder="big",
            )[
                :, :, -self.logq :
            ]  # Extract only the last `log_q` bits
            inputs = binary_matrix.reshape(bs, n * self.logq, d)
        elif self.Q > 2:
            rad = inputs / self.Q * 2 * np.pi  # convert to radians
            inputs = np.concatenate((np.cos(rad), np.sin(rad)), axis=-1)

        return super().__call__(inputs, labels)


class BinaryCodeTokenizer:
    def __init__(self, params):
        self.vocab_size = params.Q + 3
        self.separator_token = self.vocab_size - 3
        self.eos_token = self.vocab_size - 2
        self.pad_token = self.vocab_size - 1

        params.input_vocab_size = self.vocab_size
        params.output_vocab_size = 2
        params.model_output_len = 1

        pass

    def __call__(self, inputs, labels=False):
        if labels:
            return inputs.to(torch.long)

        elements = [self.tokenize(matrix) for matrix in inputs]
        lengths = [len(mat) for mat in elements]

        elements = self.pad(elements, lengths)
        lengths = torch.LongTensor(lengths)
        return dict(inputs=elements, lengths=lengths)

    def tokenize(self, matrix):
        k, mt = matrix.shape
        output_tensor = torch.empty((k, mt + 1), dtype=torch.long)
        output_tensor[:, :mt] = matrix.to(torch.long)
        output_tensor[:, mt] = self.separator_token
        output_tensor[-1, -1] = self.eos_token
        flattened = output_tensor.flatten()
        return flattened

    def pad(self, sequences, lengths):
        bs = len(sequences)
        seq_len = max(lengths)
        batch = torch.ones((bs, seq_len), dtype=torch.long, device=sequences[0].device)
        batch *= self.pad_token

        for i, seq in enumerate(sequences):
            batch[i, : len(seq)] = seq

        return batch


class BinaryCodePatchTokenizer:
    def __init__(self, params, patch_h, patch_w=1):
        """
        Initializes the tokenizer with the given patch size.
        Args:
            patch_size (int): The size of the patches (r).
        """
        assert params.Q == 2, "Binary Codes only are supported"
        self.params = params
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_size = patch_h * patch_w
        self.powers_of_two = 2 ** torch.arange(patch_h * patch_w - 1, -1, -1, dtype=int)

        params.input_vocab_size = self.vocab_size = 2 ** (patch_h * patch_w)
        params.output_vocab_size = 2

        # params.model_input_len = (k // patch_h) * (params.code_len // patch_w)
        params.model_output_len = 1
        # params.model_output_dim = 1

    def __call__(self, x, labels=False):
        if labels:
            return x.to(torch.long)
        return self.tokenize(x)

    def tokenize(self, x):
        """
        Tokenizes the input binary matrix into patches of size r by r and converts them to token IDs.
        Args:
            matrix (torch.Tensor): The input binary matrix of shape k by n.
        Returns:
            token_ids (list[int]): A list of token IDs representing the patches.
        """
        x = x[:, :, x.size(1) :]  # omit the identity in G = Ik | A
        bs, k, n = x.shape
        h, w = self.patch_h, self.patch_w
        assert k % h == 0 and n % w == 0, "Patch size does not divide matrix dimensions"
        patches = x.unfold(1, h, h).unfold(2, w, w)

        patches = patches.contiguous().view(bs, -1, h, w)

        token_ids = torch.sum(patches.view(bs, -1, h * w) * self.powers_of_two, dim=-1)

        return token_ids

    def detokenize(self, token_ids, k, n):
        """
        Reconstructs the original binary matrix from the given token IDs.
        Args:
            token_ids (list[int]): A list of token IDs representing the patches.
            k (int): The number of rows in the original matrix.
            n (int): The number of columns in the original matrix.
        Returns:
            matrix (torch.Tensor): The reconstructed binary matrix of shape k by n.
        """
        h, w = self.patch_h, self.patch_w
        assert len(token_ids) == (k // h) * (n // w), "Incorrect number of token IDs"
        patches = torch.tensor(
            [int(b) for token_id in token_ids for b in bin(token_id)[2:].zfill(h * w)]
        ).reshape(-1, h, w)
        matrix = patches.view(k // h, n // w, h, w)
        matrix = matrix.permute(0, 2, 1, 3).contiguous().view(k, n)
        return matrix


class CodeTokenizer:
    """
    Tokenizes and de-tokenizes inputs and outputs of code data
    """

    def __init__(self, params):

        self.int_base = params.Q
        self.Q = params.Q

        self.int_len = 1  # use tokens "0", "1", ..., "Q-1"

        self.symbols = [str(i) for i in range(self.int_base)]

        self.words = self.symbols + SPECIAL_WORDS

        params.vocab_size = len(self.words)

        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)

        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        logger.info(f"vocabulary: {len(self.word2id)} words")
        if len(self.word2id) < 1000:
            logger.info(f"words: {self.word2id}")


class Tokenizer(object):
    """
    Tokenizes and de-tokenizes inputs and outputs
    """

    def __init__(self, params):

        self.int_base = params.Q
        self.Q = params.Q

        self.int_len = 1

        self.symbols = [str(i) for i in range(self.int_base)]

        self.words = self.symbols + SPECIAL_WORDS

        params.input_vocab_size = params.output_vocab_size = len(self.words)

        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)

        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        logger.info(f"vocabulary: {len(self.word2id)} words")
        if len(self.word2id) < 1000:
            logger.info(f"words: {self.word2id}")

    def __call__(self, x, labels=False):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        return torch.LongTensor(np.apply_along_axis(self.encode, axis=-1, arr=x))

    def encode(self, row):
        ids = [self.word2id[str(d)] for d in row]
        return ids

    def decode(self, logits):
        device = logits.device
        ids = logits.max(dim=1)[1].cpu().numpy()
        assert ids.ndim == 2

        words = [[self.id2word[_id] for _id in seq] for seq in ids]

        b = (
            torch.LongTensor([self.decode_base(seq) for seq in words])
            .squeeze()
            .to(device)
        )
        return b

    def decode_base(self, lst):
        dim = len(lst) // self.int_len

        m = [0 for _ in range(dim)]
        for idx in range(dim):  # For each number in the sequence
            for bit in range(self.int_len):  # From high bit to low bit
                digit = lst[idx * self.int_len + bit]
                if not (digit.isdigit() or digit[0] == "-" and digit[1:].isdigit()):
                    logger.warning("Non digit tokens are not handled!")
                    continue
                m[idx] = m[idx] * self.int_base + int(digit)
        return m


class FlatCodeCompleteTokenizer(BaseCodeTokenizer):
    def __init__(self, params):
        self.Q = params.Q
        # assert self.Q == 2

    def __call__(self, inputs, labels=False):
        bs = len(inputs)
        inputs = super().__call__(inputs, labels)
        inputs = inputs.reshape(bs, -1)
        return inputs.to(torch.long)
