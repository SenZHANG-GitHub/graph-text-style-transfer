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

num_epochs = 15
checkpoint_path = 'checkpoints_gender'
restore = ''

prefix = '/home/dm/Documents/text_generation/GraphTextTransfer/data/gender/gender.short'
vocab_path = '/home/dm/Documents/text_generation/GraphTextTransfer/data/gender/vocab_gender'

train_data = {
    "batch_size": 50,
    "datasets": [
        {
            "files": "{}.train.text".format(prefix),
            "vocab_file": vocab_path,
            # Discards samples with length > 56
            "max_seq_length": 56,
            "length_filter_mode": "discard",
            # Do not append BOS/EOS tokens to the sentences
            "bos_token": "",
            "eos_token": "",
            "data_name": "x"
        },
        {
            "files": "{}.train.labels".format(prefix),
            "data_type": "int",
            "data_name": "y"
        }
    ]
}
# The val and test data have the same config with the train data, except
# for the file names
val_data = copy.deepcopy(train_data)
val_data["datasets"][0]["files"] = "{}.dev.text".format(prefix)
val_data["datasets"][1]["files"] = "{}.dev.labels".format(prefix)
test_data = copy.deepcopy(train_data)
test_data["datasets"][0]["files"] = "{}.test.text".format(prefix)
test_data["datasets"][1]["files"] = "{}.test.labels".format(prefix)

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
