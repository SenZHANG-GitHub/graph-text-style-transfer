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
Transformer encoders with multihead self attention.

For Transformer decoders with multihead cross-graph attention
* use models.cross_graph_transformer.CrossGraphTransformerFixedLengthDecoder with encoder_output=True
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.graph_multihead_attention import GraphMultiheadAttentionEncoder
from texar.core import layers
from texar.utils import transformer_attentions as attn
from texar.modules.encoders.encoder_base import EncoderBase
from texar.modules.networks.networks import FeedForwardNetwork
from texar import utils
from texar.utils.shapes import shape_list
from texar.utils.mode import is_train_mode
from texar.modules.encoders.transformer_encoders import default_transformer_poswise_net_hparams

# pylint: disable=too-many-locals, invalid-name
# pylint: disable=arguments-differ, too-many-branches, too-many-statements

__all__ = [
    "SelfGraphTransformerEncoder"
]


class SelfGraphTransformerEncoder(EncoderBase):
    """SelfGraphTransformer encoder that applies multi-head self attention for encoding
    sequences with graph structure represented by an adjacency matrix.

    See 'Texar.modules.transformer_encoders.TransformerEncoder' for details
    """
    def __init__(self, hparams=None):
        EncoderBase.__init__(self, hparams)

        with tf.variable_scope(self.variable_scope):
            if self._hparams.initializer:
                tf.get_variable_scope().set_initializer(
                    layers.get_initializer(self._hparams.initializer))

            self.graph_multihead_attention_list = []
            self.poswise_networks = []
            for i in range(self._hparams.num_blocks):
                with tf.variable_scope("layer_{}".format(i)):

                    with tf.variable_scope('attention'):
                        mh_attn = GraphMultiheadAttentionEncoder(
                            self._hparams.graph_multihead_attention)
                        self.graph_multihead_attention_list.append(mh_attn)

                        if self._hparams.dim != mh_attn.hparams.output_dim:
                            raise ValueError(
                                'The "dim" in the hparams of '
                                '"multihead_attention" should be equal to the '
                                '"dim" of GraphTransformerEncoder')

                    pw_net = FeedForwardNetwork(
                        hparams=self._hparams['poswise_feedforward'])
                    final_dim = pw_net.hparams.layers[-1]['kwargs']['units']
                    if self._hparams.dim != final_dim:
                        raise ValueError(
                            'The output dimenstion of '
                            '"poswise_feedforward" should be equal '
                            'to the "dim" of GraphTransformerEncoder.')
                    self.poswise_networks.append(pw_net)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        See 'Texar.modules.encoders.transformer_encoders.TransformerEncoder' for details
        """
        return {
            'num_blocks': 6,
            'dim': 512,
            'use_bert_config': False,
            'embedding_dropout': 0.1,
            'residual_dropout': 0.1,
            'poswise_feedforward': default_transformer_poswise_net_hparams(),
            'graph_multihead_attention': {
                'name': 'graph_multihead_attention',
                'num_units': 512,
                'num_heads': 8,
                'dropout_rate': 0.1,
                'output_dim': 512,
                'use_bias': False,
            },
            'initializer': None,
            'name': 'self_graph_transformer_encoder',
        }

    def _build(self, inputs, sequence_length, adjs, mode=None):
        """Encodes the inputs.

        Args:
            inputs: A 3D Tensor of shape `[batch_size, max_time, dim]`,
                containing the embedding of input sequences. Note that
                the embedding dimension `dim` must equal "dim" in
                :attr:`hparams`. The input embedding is typically an aggregation
                of word embedding and position embedding.
            sequence_length: A 1D Tensor of shape `[batch_size]`. Input tokens
                beyond respective sequence lengths are masked out
                automatically.
            adjs: A 3D Tensor of shape `[batch_size, max_time, max_time]`,
                containing the adjacency matrices of input sequences
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`,
                including `TRAIN`, `EVAL`, and `PREDICT`. Used to toggle
                dropout.
                If `None` (default), :func:`texar.global_mode` is used.

        Returns:
            A Tensor of shape `[batch_size, max_time, dim]` containing the
            encoded vectors.
        """
        # Multiply input embedding with the sqrt of its dimension for
        # normalization

        adj_masks = 1 - tf.cast(tf.equal(adjs, 0), dtype=tf.float32)

        inputs_padding = 1 - tf.sequence_mask(
            sequence_length, tf.shape(inputs)[1], dtype=tf.float32)
        if self._hparams.use_bert_config:
            ignore_padding = attn.attention_bias_ignore_padding(
                inputs_padding, bias_value=-1e4)
        else:
            ignore_padding = attn.attention_bias_ignore_padding(
                inputs_padding)
        encoder_self_attention_bias = ignore_padding

        input_embedding = inputs # shape (batch_size, max_time, dim)

        if self._hparams.use_bert_config:
            x = layers.layer_normalize(input_embedding)
            x = tf.layers.dropout(x,
                                  rate=self._hparams.embedding_dropout,
                                  training=is_train_mode(mode))
        else:
            x = tf.layers.dropout(input_embedding,
                                  rate=self._hparams.embedding_dropout,
                                  training=is_train_mode(mode))

        # Just to keep consistent with BERT, actually makes no difference
        if self._hparams.use_bert_config:
            pad_remover = None
        else:
            pad_remover = utils.transformer_utils.PadRemover(inputs_padding)

        for i in range(self._hparams.num_blocks):
            with tf.variable_scope("layer_{}".format(i)):
                graph_multihead_attention = self.graph_multihead_attention_list[i]

                # trivial difference between BERT and original Transformer
                if self._hparams.use_bert_config:
                    _queries_input = x
                else:
                    _queries_input = layers.layer_normalize(x)

                attention_output = graph_multihead_attention(
                    queries=_queries_input,
                    memory=_queries_input,
                    adj_masks=adj_masks,
                    memory_attention_bias=encoder_self_attention_bias,
                    mode=mode,
                )
                attention_output = tf.layers.dropout(
                    attention_output,
                    rate=self._hparams.residual_dropout,
                    training=is_train_mode(mode),
                )
                # attention_output: weighted sum of V of memory with weights determined by querying keys of memory
                x = x + attention_output
                with tf.variable_scope('output'):
                    if self._hparams.use_bert_config:
                        x = layers.layer_normalize(x)
                        y = x
                    else:
                        y = layers.layer_normalize(x)

                poswise_network = self.poswise_networks[i]
                with tf.variable_scope(poswise_network.variable_scope):
                    original_shape = shape_list(y)
                    y = tf.reshape(y, [-1, self._hparams.dim])
                    if pad_remover:
                        y = tf.expand_dims(pad_remover.remove(y), axis=0)
                        # [1, batch_size*seq_length, hidden_dim]
                    layer_output = poswise_network(y, mode=mode)
                    sub_output = tf.layers.dropout(
                        layer_output,
                        rate=self._hparams.residual_dropout,
                        training=is_train_mode(mode)
                    )
                    if pad_remover:
                        sub_output = tf.reshape(pad_remover.restore(tf.squeeze(\
                            sub_output, axis=0)), original_shape \
                        )
                    else:
                        sub_output = tf.reshape(sub_output, original_shape)

                    x = x + sub_output
                    if self._hparams.use_bert_config:
                        x = layers.layer_normalize(x)

        if not self._hparams.use_bert_config:
            x = layers.layer_normalize(x)

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return x
