# Copyright 2019 The Texar Authors. All Rights Reserved.
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
"""
Data class that supports reading TFRecord data and data type converting.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from texar.data.data import dataset_utils as dsutils
from texar.data.data.data_base import DataBase
from texar.data.data.mono_text_data import MonoTextData
from texar.data.data.tfrecord_data import TFRecordData

from utils_data.data_decoders_with_numpy import TFRecordDataNumpyDecoder


# pylint: disable=invalid-name, arguments-differ, not-context-manager

__all__ = [
    "_default_tfrecord_numpy_dataset_hparams",
    "TFRecordNumpyData"
]

def _default_tfrecord_numpy_dataset_hparams():
    """Returns hyperparameters of a TFRecord dataset with numpy support with default values.

    See :meth:`texar.data.TFRecordNumpyData.default_hparams` for details.
    """
    return {
        "files": [],
        "feature_original_types": {},
        "feature_convert_types": {},
        "image_options": {},
        "numpy_options": {},
        "compression_type": None,
        "other_transformations": [],
        "num_shards": None,
        "shard_id": None,
        "data_name": None,
        "@no_typecheck": [
            "files",
            "feature_original_types",
            "feature_convert_types",
            "image_options",
            "numpy_options"
        ],
    }

class TFRecordNumpyData(TFRecordData):
    """TFRecord data which loads and processes TFRecord files with numpy support.

    See `texar.data.TFRecordData` for details
    """

    def __init__(self, hparams):
        TFRecordData.__init__(self, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dicitionary of default hyperparameters.

        See `texar.data.TFRecordData.default_hparams` for details
        """
        hparams = TFRecordData.default_hparams()
        hparams["name"] = "tfrecord_data"
        hparams.update({
            "dataset": _default_tfrecord_numpy_dataset_hparams()
        })
        return hparams

    @staticmethod
    def _make_processor(dataset_hparams, data_spec, chained=True,
                        name_prefix=None):
        # Create data decoder
        decoder = TFRecordDataNumpyDecoder(
            feature_original_types=dataset_hparams.feature_original_types,
            feature_convert_types=dataset_hparams.feature_convert_types,
            image_options=dataset_hparams.image_options,
            numpy_options=dataset_hparams.numpy_options)
        # Create other transformations
        data_spec.add_spec(decoder=decoder)
        # pylint: disable=protected-access
        other_trans = MonoTextData._make_other_transformations(
            dataset_hparams["other_transformations"], data_spec)

        data_spec.add_spec(name_prefix=name_prefix)

        if chained:
            chained_tran = dsutils.make_chained_transformation(
                [decoder] + other_trans)
            return chained_tran, data_spec
        else:
            return decoder, other_trans, data_spec

