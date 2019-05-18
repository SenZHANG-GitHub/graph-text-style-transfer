# -*- coding: utf-8 -*-
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
Helper functions and classes for decoding text data which are used after
reading raw text data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import data_decoder

from texar.data.vocabulary import SpecialTokens
from texar.utils import dtypes
from texar.hyperparams import HParams
from texar.data.data_decoders import TFRecordDataDecoder


# pylint: disable=too-many-instance-attributes, too-many-arguments,
# pylint: disable=no-member, invalid-name

__all__ = [
    "TFRecordDataNumpyDecoder",
]

class TFRecordDataNumpyDecoder(TFRecordDataDecoder):
    """A data decoder that decodes a TFRecord file, e.g., the
    TFRecord file with numpy support

    See 'texar.data.data_decoders.TFRecordDataDecoder' for details
    """

    def __init__(self,
                 feature_original_types,
                 feature_convert_types,
                 image_options,
                 numpy_options):
        TFRecordDataDecoder.__init__(self, feature_original_types, feature_convert_types, image_options)
        self._numpy_options = numpy_options

    def _decode_numpy_ndarray_str_byte(self,
                                       numpy_option_feature,
                                       decoded_data):
        numpy_key = numpy_option_feature.get('numpy_ndarray_name')
        if numpy_key is None:
            return

        shape = numpy_option_feature.get('shape')
        dtype = numpy_option_feature.get('dtype')
        dtype = dtypes.get_tf_dtype(dtype)

        numpy_byte = decoded_data.get(numpy_key)
        numpy_ndarray = tf.decode_raw(numpy_byte, dtype)
        numpy_ndarray = tf.reshape(numpy_ndarray, shape)

        decoded_data[numpy_key] = numpy_ndarray


    def decode(self, data, items):
        """Decodes the data to return the tensors specified by the list of
        items with numpy support.

        Args:
            data: The TFRecord data(serialized example) to decode.
            items: A list of strings, each of which is the name of the resulting
                tensors to retrieve.

        Returns:
            A list of tensors, each of which corresponds to each item.
        """
        # pylint: disable=too-many-branches
        feature_description = dict()
        for key, value in  self._feature_original_types.items():
            shape = []
            if len(value) == 3:
                if isinstance(value[-1], int):
                    shape = [value[-1]]
                elif isinstance(value[-1], list):
                    shape = value
            if len(value) < 2 or value[1] == 'FixedLenFeature':
                feature_description.update(
                    {key: tf.FixedLenFeature(
                        shape,
                        dtypes.get_tf_dtype(value[0]))})
            elif value[1] == 'VarLenFeature':
                feature_description.update(
                    {key: tf.VarLenFeature(
                        dtypes.get_tf_dtype(value[0]))})
        decoded_data = tf.parse_single_example(data, feature_description)

        # Handle TFRecord containing images
        if isinstance(self._image_options, dict):
            self._decode_image_str_byte(
                self._image_options,
                decoded_data)
        elif isinstance(self._image_options, HParams):
            self._decode_image_str_byte(
                self._image_options.todict(),
                decoded_data)
        elif isinstance(self._image_options, list):
            _ = list(map(
                lambda x: self._decode_image_str_byte(x, decoded_data),
                self._image_options))

        # Handle TFRecord containing numpy.ndarray
        if isinstance(self._numpy_options, dict):
            self._decode_numpy_ndarray_str_byte(
                self._numpy_options,
                decoded_data)
        elif isinstance(self._numpy_options, HParams):
            self._decode_numpy_ndarray_str_byte(
                self._numpy_options.todict(),
                decoded_data)

        # Convert Dtypes
        for key, value in self._feature_convert_types.items():
            from_type = decoded_data[key].dtype
            to_type = dtypes.get_tf_dtype(value)
            if from_type is to_type:
                continue
            elif to_type is tf.string:
                decoded_data[key] = tf.dtypes.as_string(decoded_data[key])
            elif from_type is tf.string:
                decoded_data[key] = tf.string_to_number(
                    decoded_data[key], to_type)
            else:
                decoded_data[key] = tf.cast(
                    decoded_data[key], to_type)
        outputs = decoded_data
        return [outputs[item] for item in items]
