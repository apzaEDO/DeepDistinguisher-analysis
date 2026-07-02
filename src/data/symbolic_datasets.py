# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from src.data.distinguisher_datasets import GoppaDistAllDataset
import logging

import torch


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class GoppaSymDistDataset(GoppaDistAllDataset):
    def __getitem__(self, index):
        """
        Open an HDF5 file and return a sample from the dataset at the specified index.
        Args:
        index (int): Index of the sample to return.
        """
        label = index % 2
        ix = index // 2
        data = self.datasources[label].read_item(ix)

        G = data[self.key]
        G = torch.from_numpy(G)

        return G, torch.Tensor([label])

    def collate_fn(self, elements):
        X, labels = zip(*elements)

        labels = torch.stack(labels)

        inputs = self.tokenizer(X)
        labels = self.tokenizer(labels, labels=True)

        return dict(**inputs, labels=labels)
