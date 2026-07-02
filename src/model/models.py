# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LRNet(nn.Module):
    def __init__(self, params):
        super(LRNet, self).__init__()

        self.w = params.code_len
        self.h = params.code_len - params.m_alt * params.t_alt

        self.fc = nn.Linear(self.w * self.h, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs, labels):
        x = inputs.view(-1, self.w * self.h)

        logits = self.fc(x)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return dict(output=(logits > 0).to(int), loss=loss)


class NNet(nn.Module):
    def __init__(self, params):
        super(NNet, self).__init__()

        self.w = params.code_len
        self.h = params.code_len - params.m_alt * params.t_alt

        # Define a fully connected layer that outputs the logit
        self.d = params.enc_emb_dim

        self.fcs = nn.ModuleList(
            [nn.Linear(self.w * self.h, self.d)]
            + [nn.Linear(self.d, self.d) for _ in range(params.n_enc_layers - 2)]
        )
        self.head = nn.Linear(
            self.d, 1
        )  # Adjust size according to your input dimensions
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs, labels):

        x = inputs.view(-1, self.w * self.h)

        for linear in self.fcs:
            x = F.relu(linear(x))

        # Output layer that returns the logit
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return dict(output=(logits > 0).to(int), loss=loss)


class ConvNet(nn.Module):
    def __init__(self, params):
        super(ConvNet, self).__init__()

        self.w = params.m_alt * params.t_alt
        self.h = params.code_len - self.w

        # Define the first convolutional layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # Define the second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Define a max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Define a fully connected layer that outputs the logit
        self.d = 256 * (self.h // 16) * (self.w // 16)
        print(self.d)
        self.fcs = nn.ModuleList(
            [nn.Linear(self.d, self.d) for _ in range(params.n_enc_layers - 1)]
        )
        self.head = nn.Linear(
            self.d, 1
        )  # Adjust size according to your input dimensions
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, G, labels):
        x = G.unsqueeze(1)
        # Apply the first convolutional layer followed by ReLU and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = x.view(-1, self.d)
        for linear in self.fcs:
            x = F.relu(linear(x))

        # Output layer that returns the logit
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return dict(output=(logits > 0).to(int), loss=loss)


class TLinear(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        tx = input.swapaxes(2, 1)
        return F.linear(tx, self.weight, self.bias).swapaxes(2, 1)


class TAttn(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        x = input
        tx = x.swapaxes(2, 1)
        xtx = F.softmax(x @ tx, dim=-1)
        weight = xtx @ self.weight
        return weight @ x


class LRNNet(nn.Module):
    def __init__(self, params):
        super(LRNNet, self).__init__()

        self.w = params.code_len
        self.h = params.code_len - params.m_alt * params.t_alt

        # Define a fully connected layer that outputs the logit
        self.d = params.enc_emb_dim

        self.proj = nn.Linear(self.w, self.d)

        self.fcs = nn.ModuleList(
            [nn.Linear(self.d, self.d) for _ in range(params.n_enc_layers)]
        )
        self.tfcs = nn.ModuleList(
            [TLinear(self.h, self.h) for _ in range(params.n_enc_layers)]
        )
        self.head = nn.Linear(
            self.d, 1
        )  # Adjust size according to your input dimensions
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs, labels):

        x = self.proj(inputs)

        for left, right in zip(self.tfcs, self.fcs):
            x = F.gelu(right(F.gelu(left(x)))) + x

        x = torch.max(x, dim=1)[0]
        # Output layer that returns the logit
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return dict(output=(logits > 0).to(int), loss=loss)


class AttNet(LRNNet):
    def __init__(self, params):
        super().__init__(params)
        self.tfcs = nn.ModuleList(
            [TAttn(self.h, self.h) for _ in range(params.n_enc_layers)]
        )
