# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from src.data.datasource import GoppaDataDirSource
from src.data.distinguisher_datasets import CodeDistDataset
import numpy as np
import logging


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class CodeIdentifierDataset(CodeDistDataset):
    def __init__(self, tokenizer, params, size):
        self._set_data_keys()
        super().__init__(tokenizer, params, size)

    def _set_data_keys(self):
        raise NotImplementedError()

    def _set_datasources(self, params, size):
        if self.code in self.GENERATORS_CLS:
            self.code_generator = self.GENERATORS_CLS[self.code](params)
        else:
            raise NotImplementedError(f"task {params.task} not supported")

        self.datasources = [self.code_generator]

    def _set_processed_data_dims(self, params, tokenizer):
        params.model_input_dim = params.m_alt * params.t_alt
        params.model_input_len = params.code_len - params.m_alt * params.t_alt
        params.model_output_dim = params.model_input_dim
        params.decoder_output_dim = params.m_alt
        params.decoder_output_len = params.t_alt if params.pred_what == "all" else 1

    def __getitem__(self, index):
        """
        Open an HDF5 file and return a sample from the dataset at the specified index.
        Args:
        index (int): Index of the sample to return.
        """
        isource = index % len(self.datasources)
        ix = index // len(self.datasources)
        data = self.datasources[isource].read_item(ix)

        data["inputs"] = data.pop(self.in_key)
        if hasattr(self, "out_key") and self.out_key is not None:
            data["labels"] = data.pop(self.out_key)
        else:
            data["labels"] = data["inputs"]

        return data


class GoppaCompleteDataset(CodeIdentifierDataset):
    def __init__(self, tokenizer, params, size):
        super().__init__(tokenizer, params, size)

        self.n_masked = params.n_masked

    def _set_processed_data_dims(self, params, tokenizer):
        params.model_input_dim = params.m_alt * params.t_alt
        params.model_input_len = params.k = params.code_len - params.m_alt * params.t_alt
        params.model_output_dim = params.model_input_dim
        params.model_output_len = params.model_input_len 
        params.output_vocab_size = params.Q

        self.fill_value = -1

    def _set_datasources(self, params, size):
        self.datasources = [GoppaDataDirSource(params, size, [self.in_key])]
        self.size = sum(map(len, self.datasources))
        if params.repset_size > 0:
            miniset_size = params.repset_size
            self.datasources.append(
                GoppaDataDirSource(params, miniset_size, [self.in_key])
            )
            self.size = 2 * len(self.datasources[0])

    def _set_data_keys(self):
        self.in_key = "G"

    def collate_fn(self, elements):
        data = {key: self.tokenizer([d[key] for d in elements]) for key in elements[0]}

        return data

    def __getitem__(self, index):
        """
        Open the HDF5 file and return a sample from the dataset at the specified index.
        Args:
        index (int): Index of the sample to return.
        """

        data = super().__getitem__(index)
        G = data["inputs"]
        # G and G_masked should be in shape k by n-k
        if G.shape[-1] == self.params.code_len:
            G = G[:, self.params.k :]

        data = self._mask(G, self.n_masked)

        return data

    def _mask(self, matrix, num_masked_elements=0.15):
        """
        Create a mask for the given matrix where the selected entries are set to -1.

        Parameters:
            matrix (np.array): A numpy array of shape (k, n).
            mask_prob (float): Probability of each element being masked. Default is 0.15.

        Returns:
            np.array: A masked numpy array of the same shape as the input.
        """
        assert num_masked_elements >= 0, "num_masked_elements should be non negative"
        if num_masked_elements >= 1:
            flat_matrix = matrix.flatten()

            total_elements = flat_matrix.size
            mask_indices = np.random.choice(
                total_elements, int(num_masked_elements), replace=False
            )

            flat_mask = np.zeros(total_elements, dtype=bool)
            flat_mask[mask_indices] = True

            # Reshape the flat mask to the original matrix shape
            mask = flat_mask.reshape(matrix.shape)

        else:  # a probability
            mask = np.random.random(matrix.shape) < num_masked_elements

        # Create the masked matrix by setting masked positions to a neutral value (e.g., 0)
        masked_matrix = np.where(mask, self.fill_value, matrix)

        return dict(inputs=masked_matrix, labels=matrix, masks=mask)


class FlatGoppaCompleteDataset(GoppaCompleteDataset):
    def _set_processed_data_dims(self, params, tokenizer):
        params.model_input_dim = 1
        params.k = params.code_len - params.m_alt * params.t_alt
        params.model_input_len = params.k * params.m_alt * params.t_alt
        params.model_output_len = params.model_input_len
        params.model_output_dim = params.Q
        params.tokenize = True
        params.input_vocab_size = params.Q + 2
        self.fill_value = params.Q
        params.positional_encoding_nrows = params.code_len - params.m_alt * params.t_alt


class GoppaCorrectDataset(GoppaCompleteDataset):

    def _mask(self, matrix, num_masked_elements=0.15):
        """
        Create a mask for the given matrix where the selected entries are set to -1.

        Parameters:
            matrix (np.array): A numpy array of shape (k, n).
            mask_prob (float): Probability of each element being masked. Default is 0.15.

        Returns:
            np.array: A masked numpy array of the same shape as the input.
        """
        assert num_masked_elements >= 0, "num_masked_elements should be non negative"
        if num_masked_elements >= 1:
            flat_matrix = matrix.flatten()

            total_elements = flat_matrix.size
            mask_indices = np.random.choice(
                total_elements, int(num_masked_elements), replace=False
            )

            flat_mask = np.ones(total_elements, dtype=bool)
            flat_mask[mask_indices] = False

            # Reshape the flat mask to the original matrix shape
            mask = flat_mask.reshape(matrix.shape)

        else:  # a probability
            mask = np.random.random(matrix.shape) > num_masked_elements

        # Create the masked matrix by setting masked positions to a neutral value (e.g., 0)
        faulty_matrix = np.where(mask, matrix, 1 - matrix)

        return dict(inputs=faulty_matrix, labels=mask)
