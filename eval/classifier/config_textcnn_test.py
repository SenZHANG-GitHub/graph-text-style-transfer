# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sentence convolutional classifier config.

This is (approximately) the config of the paper:
(Kim) Convolutional Neural Networks for Sentence Classification
  https://arxiv.org/pdf/1408.5882.pdf
"""

# pylint: disable=invalid-name, too-few-public-methods, missing-docstring

import copy

# Word embedding
emb = {
    "dim": 300
}

# Classifier
clas = {
    "num_conv_layers": 1,
    "filters": 100,
    "kernel_size": [3, 4, 5],
    "conv_activation": "relu",
    "pooling": "MaxPooling1D",
    "num_dense_layers": 0,
    "dropout_conv": [1],
    "dropout_rate": 0.5,
    "num_classes": 2
}

# Optimization
# Just use the default config, e.g., Adam Optimizer
opt = {}
