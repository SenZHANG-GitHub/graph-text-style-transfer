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
"""
Data consisting of multiple aligned parts.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy

import tensorflow as tf

from texar.hyperparams import HParams
from texar.utils import utils
from texar.utils.dtypes import is_str, is_callable
from texar.data.data.text_data_base import TextDataBase
from texar.data.data.scalar_data import ScalarData
from texar.data.data.mono_text_data import _default_mono_text_dataset_hparams
from texar.data.data.scalar_data import _default_scalar_dataset_hparams
from texar.data.data.mono_text_data import MonoTextData
from texar.data.data_utils import count_file_lines
from texar.data.data import dataset_utils as dsutils
from texar.data.vocabulary import Vocab, SpecialTokens
from texar.data.embedding import Embedding
from texar.data.data.multi_aligned_data import _default_dataset_hparams
from texar.data.data.multi_aligned_data import MultiAlignedData

from utils_data.tfrecord_data_with_numpy import TFRecordNumpyData
from utils_data.tfrecord_data_with_numpy import _default_tfrecord_numpy_dataset_hparams


# pylint: disable=invalid-name, arguments-differ
# pylint: disable=protected-access, too-many-instance-attributes

__all__ = [
    "_default_dataset_numpy_hparams",
    "MultiAlignedNumpyData"
]

class _DataTypes(object): # pylint: disable=no-init, too-few-public-methods
    """Enumeration of data types.
    """
    TEXT = "text"
    INT = "int"
    FLOAT = "float"
    TF_RECORD = "tf_record"

def _is_text_data(data_type):
    return data_type == _DataTypes.TEXT
def _is_scalar_data(data_type):
    return data_type == _DataTypes.INT or data_type == _DataTypes.FLOAT
def _is_tfrecord_data(data_type):
    return data_type == _DataTypes.TF_RECORD

def _default_dataset_numpy_hparams(data_type=None):
    """Returns hyperparameters of a dataset with default values.

    See :meth:`texar.data.MultiAlignedData.default_hparams` for details.
    """
    if not data_type or _is_text_data(data_type):
        hparams = _default_mono_text_dataset_hparams()
        hparams.update({
            "data_type": _DataTypes.TEXT,
            "vocab_share_with": None,
            "embedding_init_share_with": None,
            "processing_share_with": None,
        })
    elif _is_scalar_data(data_type):
        hparams = _default_scalar_dataset_hparams()
    elif _is_tfrecord_data(data_type):
        # use numpy-supported tfrecord hparams here
        hparams = _default_tfrecord_numpy_dataset_hparams()
        hparams.update({
            "data_type": _DataTypes.TF_RECORD,
        })
    return hparams

class MultiAlignedNumpyData(MultiAlignedData, TextDataBase):
    """Data consisting of multiple aligned parts.

    See `texar.data.MultiAlignedData` for details.
    """
    def __init__(self, hparams):
        TextDataBase.__init__(self, hparams)
        # Defaultizes hparams of each dataset
        datasets_hparams = self._hparams.datasets
        defaultized_datasets_hparams = []
        for ds_hpms in datasets_hparams:
            data_type = ds_hpms.get("data_type", None)
            defaultized_ds_hpms = HParams(ds_hpms,
                                          _default_dataset_numpy_hparams(data_type))
            defaultized_datasets_hparams.append(defaultized_ds_hpms)
        self._hparams.datasets = defaultized_datasets_hparams

        with tf.name_scope(self.name, self.default_hparams()["name"]):
            self._make_data()

    @staticmethod
    def _make_processor(dataset_hparams, data_spec, name_prefix):
        processors = []
        for i, hparams_i in enumerate(dataset_hparams):
            data_spec_i = data_spec.get_ith_data_spec(i)

            data_type = hparams_i["data_type"]
            if _is_text_data(data_type):
                tgt_proc_hparams = hparams_i
                proc_shr = hparams_i["processing_share_with"]
                if proc_shr is not None:
                    tgt_proc_hparams = copy.copy(dataset_hparams[proc_shr])
                    try:
                        tgt_proc_hparams["variable_utterance"] = \
                                hparams_i["variable_utterance"]
                    except TypeError:
                        tgt_proc_hparams.variable_utterance = \
                                hparams_i["variable_utterance"]

                processor, data_spec_i = MonoTextData._make_processor(
                    tgt_proc_hparams, data_spec_i)
            elif _is_scalar_data(data_type):
                processor, data_spec_i = ScalarData._make_processor(
                    hparams_i, data_spec_i, name_prefix='')
            elif _is_tfrecord_data(data_type):
                processor, data_spec_i = TFRecordNumpyData._make_processor(
                    hparams_i, data_spec_i, name_prefix='')
            else:
                raise ValueError("Unsupported data type: %s" % data_type)

            processors.append(processor)
            data_spec.set_ith_data_spec(i, data_spec_i, len(dataset_hparams))

        tran_fn = dsutils.make_combined_transformation(
            processors, name_prefix=name_prefix)

        data_spec.add_spec(name_prefix=name_prefix)

        return tran_fn, data_spec
