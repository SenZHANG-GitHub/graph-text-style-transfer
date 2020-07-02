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
Transformer decoders with multihead cross-graph attention.

CrossGraphTransformerFixedLengthDecoder: 
* forked from models.self_graph_transofrmer.SelfGraphTransformerEncoder
* change the attention from self-graph attention to cross-graph attention
* implemented in encoder manner and add an output layer + Gumbel-softmax to get vocabulary probabilities for decoding
* use encoder_output (boolean) to control whether it serves as a cross-graph transformer encoder or a fixed-length decoder

CrossGraphTransformerSequentialDecoder:
* forked from Texar.modules.decoders.transformer_decoder.TransformerDecoder
* introduce adjacency matrix for neighbor-wise attention
* change multihead_attention to graph_multihead_attention

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
from tensorflow.contrib.seq2seq import Decoder as TFDecoder
from tensorflow.contrib.seq2seq import dynamic_decode
from tensorflow.contrib.distributions import RelaxedOneHotCategorical as GumbelSoftmax

from texar import utils
from texar.core import layers
from texar.utils.shapes import shape_list
from texar.utils.mode import is_train_mode
from texar.utils import transformer_attentions as attn
from texar.module_base import ModuleBase
from texar.modules.encoders.encoder_base import EncoderBase
from texar.modules.networks.networks import FeedForwardNetwork
from texar.modules.decoders import tf_helpers as tx_helper
from texar.modules.decoders.rnn_decoder_base import _make_output_layer
from texar.modules.encoders.transformer_encoders import default_transformer_poswise_net_hparams
from texar.modules.decoders.transformer_decoders import TransformerDecoder
from texar.modules.decoders.transformer_decoders import TransformerDecoderOutput

from models.graph_multihead_attention import GraphMultiheadAttentionEncoder

# pylint: disable=too-many-locals, invalid-name
# pylint: disable=arguments-differ, too-many-branches, too-many-statements

__all__ = [
    "CrossGraphTransformerFixedLengthDecoderOutput",
    "CrossGraphTransformerFixedLengthDecoder",
    "CrossGraphTransformerSequentialDecoder"
]


class CrossGraphTransformerFixedLengthDecoderOutput(collections.namedtuple(
    "CrossGraphTransformerFixedLengthDecoderOutput",
    ("logits", "sample_id", "probs"))):
    """The output of :class:'CrossGraphTransformerFixedLengthDecoder'.
    
    See 'texar.modules.decoders.transformer_decoders.TransformerDecoder' for details

    probs: If encoder_output is True: A float Tensor of shape 
            '[batch_size, max_time, vocab_size]' containing the probabilities
           If encoder_output is False: probs = ''
           
    """


class CrossGraphTransformerFixedLengthDecoder(EncoderBase):
    """CrossGraphTransformer encoder/decoder that applies multi-head cross-graph attention for encoding sequences with graph structure represented by an adjacency matrix.

    See 'Texar.modules.encoders.transformer_encoders.TransformerEncoder' for details
    See 'Texar.modules.decoders.transformer_decoders.TransformerDecoder' for details

    Implemented in the encoder manner. But we can add an output layer to each node to get vocabulary probabilities
    """
    def __init__(self, 
                 vocab_size=None,
                 output_layer=None,
                 tau = None,
                 hparams=None):
        EncoderBase.__init__(self, hparams)

        with tf.variable_scope(self.variable_scope):
            if self._hparams.initializer:
                tf.get_variable_scope().set_initializer(
                    layers.get_initializer(self._hparams.initializer))
            
            # Make the output layer
            self._output_layer, self._vocab_size = _make_output_layer(
                output_layer, vocab_size, self._hparams.output_layer_bias,
                self.variable_scope
            )

            # Make attention and poswise networks
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
                                '"dim" of CrossGraphTransformerFixedLengthDecoder')

                    pw_net = FeedForwardNetwork(
                        hparams=self._hparams['poswise_feedforward'])
                    final_dim = pw_net.hparams.layers[-1]['kwargs']['units']
                    if self._hparams.dim != final_dim:
                        raise ValueError(
                            'The output dimenstion of '
                            '"poswise_feedforward" should be equal '
                            'to the "dim" of CrossGraphTransformerFixedLengthDecoder.')
                    self.poswise_networks.append(pw_net)
            
            self._helper = None
            self._tau = tau

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
            'name': 'cross_graph_transformer_fixed_length_decoder',
            'embedding_tie': True,
            'output_layer_bias': False,
            'max_decoding_length': int(1e10),
        }

    def _build(self, inputs, memory, sequence_length, memory_sequence_length, adjs, encoder_output, mode=None):
        """Encodes the inputs.

        Args:
            inputs: A 3D Tensor of shape `[batch_size, max_time, dim]`,
                containing the embedding of input sequences. Note that
                the embedding dimension `dim` must equal "dim" in
                :attr:`hparams`. The input embedding is typically an aggregation
                of word embedding and position embedding.
            memory: A 3D Tensor of shape `[batch_size, memory_max_time, dim]`, 
                containing the embedding of memory sequences. Note that
                the embedding dimension `dim` must equal "dim" in
                :attr:`hparams`. The input embedding is typically an aggregation
                of word embedding and position embedding.
            sequence_length: A 1D Tensor of shape `[batch_size]`. Input tokens
                beyond respective sequence lengths are masked out
                automatically.
            sequence_length: A 1D Tensor of shape `[batch_size]`. Memory tokens
                beyond respective sequence lengths are masked out
                automatically.
            adjs: A 3D Tensor of shape `[batch_size, max_time, max_time]`,
                containing the adjacency matrices of input sequences
            encoder_output: bool. True: return encoder-like embeddings. False: return CrossGraphTransformerDecoderOutput. 
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`,
                including `TRAIN`, `EVAL`, and `PREDICT`. Used to toggle
                dropout.
                If `None` (default), :func:`texar.global_mode` is used.

        Returns:
            A Tensor of shape `[batch_size, max_time, dim]` containing the
            encoded vectors.
        """
        # Get adjacency masks from adjs
        adj_masks = 1 - tf.cast(tf.equal(adjs, 0), dtype=tf.float32)

        # Multiply input embedding with the sqrt of its dimension for
        # normalization
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
                    memory=memory,
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

        if encoder_output:
            return x
        
        logits = self._output_layer(x)
        sample_ids = tf.to_int32(tf.argmax(logits, axis=-1))
        probs = ''
        # probs = GumbelSoftmax(self._tau, logits=logits).sample()
        # probs = tf.nn.softmax(logits / self._tau) # vanilla softmax

        rets = CrossGraphTransformerFixedLengthDecoderOutput(
            logits = logits,
            sample_id = sample_ids,
            probs = probs
        )

        return rets


class CrossGraphTransformerSequentialDecoder(TransformerDecoder, ModuleBase, TFDecoder):
    """Transformer decoder that applies cross-graph multi-head self-attention for
    sequence decoding.

    See 'Texar.modules.decoders.transformer_decoders.TransformerDecoder' for details

    This Decoder can be used almost in the same way as TransformerDecoder, except for the additional adjs
    """

    def __init__(self,
                 vocab_size=None,
                 output_layer=None,
                 hparams=None):
        ModuleBase.__init__(self, hparams)
        self._hparams.add_hparam('multihead_attention', self._hparams.graph_multihead_attention)

        with tf.variable_scope(self.variable_scope):
            if self._hparams.initializer:
                tf.get_variable_scope().set_initializer(
                    layers.get_initializer(self._hparams.initializer))

            # Make the output layer
            self._output_layer, self._vocab_size = _make_output_layer(
                output_layer, vocab_size, self._hparams.output_layer_bias,
                self.variable_scope)

            # Make attention and poswise networks
            self.multihead_attentions = {
                'self_att': [],
                'encdec_att': []
            }
            self.poswise_networks = []
            for i in range(self._hparams.num_blocks):
                layer_name = 'layer_{}'.format(i)
                with tf.variable_scope(layer_name):
                    with tf.variable_scope("self_attention"):
                        multihead_attention = GraphMultiheadAttentionEncoder(
                            self._hparams.graph_multihead_attention)
                        self.multihead_attentions['self_att'].append(
                            multihead_attention)

                    if self._hparams.dim != \
                            multihead_attention.hparams.output_dim:
                        raise ValueError('The output dimenstion of '
                                         'MultiheadEncoder should be equal '
                                         'to the dim of TransformerDecoder')

                    with tf.variable_scope('encdec_attention'):
                        multihead_attention = GraphMultiheadAttentionEncoder(
                            self._hparams.graph_multihead_attention)
                        self.multihead_attentions['encdec_att'].append(
                            multihead_attention)

                    if self._hparams.dim != \
                            multihead_attention.hparams.output_dim:
                        raise ValueError('The output dimenstion of '
                                         'MultiheadEncoder should be equal '
                                         'to the dim of TransformerDecoder')

                    pw_net = FeedForwardNetwork(
                        hparams=self._hparams['poswise_feedforward'])
                    final_dim = pw_net.hparams.layers[-1]['kwargs']['units']
                    if self._hparams.dim != final_dim:
                        raise ValueError(
                            'The output dimenstion of '
                            '"poswise_feedforward" should be equal '
                            'to the "dim" of TransformerDecoder.')
                    self.poswise_networks.append(pw_net)

            # Built in _build()
            self.context = None
            self.context_sequence_length = None
            self.embedding = None
            self._helper = None
            self._cache = None

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        See 'Texar.modules.decoders.transformer_decoders.TransformerDecoder' for details
        """
        return {
            "num_blocks": 6,
            "dim": 512,
            "embedding_tie": True,
            "output_layer_bias": False,
            "max_decoding_length": int(1e10),
            "embedding_dropout": 0.1,
            "residual_dropout": 0.1,
            "poswise_feedforward": default_transformer_poswise_net_hparams(),
            'graph_multihead_attention': {
                'name': 'graph_multihead_attention',
                'num_units': 512,
                'num_heads': 8,
                'dropout_rate': 0.1,
                'output_dim': 512,
                'use_bias': False,
            },
            "initializer": None,
            "name": "cross_graph_transformer_sequential_decoder",
        }
    

    def _build(self,  # pylint: disable=arguments-differ, too-many-statements
               decoding_strategy='train_greedy',
               inputs=None,
               adjs=None,
               memory=None,
               memory_sequence_length=None,
               memory_attention_bias=None,
               beam_width=None,
               length_penalty=0.,
               start_tokens=None,
               end_token=None,
               context=None,
               context_sequence_length=None,
               softmax_temperature=None,
               max_decoding_length=None,
               impute_finished=False,
               embedding=None,
               helper=None,
               mode=None):
        """Performs decoding.

        See 'Texar.modules.decoders.transformer_decoders.TransformerDecoder' for details

        adjs: A 3D Tensor of shape `[batch_size, max_time, max_time]`,
                containing the adjacency matrices of input sequences
        """
        # Get adjacency masks from adjs
        self.adj_masks = 1 - tf.cast(tf.equal(adjs, 0), dtype=tf.float32)

        if memory is not None:
            if memory_attention_bias is None:
                if memory_sequence_length is None:
                    raise ValueError(
                        "`memory_sequence_length` is required if "
                        "`memory_attention_bias` is not given.")

                enc_padding = 1 - tf.sequence_mask(
                    memory_sequence_length, shape_list(memory)[1],
                    dtype=tf.float32)
                memory_attention_bias = attn.attention_bias_ignore_padding(
                    enc_padding)

        # record the context, which will be used in step function
        # for dynamic_decode
        if context is not None:
            start_tokens = context[:, 0]
            self.context = context[:, 1:]
            self.context_sequence_length = context_sequence_length - 1
        else:
            self.context = None

        self.embedding = embedding

        if helper is None and beam_width is None and \
                decoding_strategy == 'train_greedy':  # Teacher-forcing

            decoder_self_attention_bias = (
                attn.attention_bias_lower_triangle(
                    shape_list(inputs)[1]))

            decoder_output = self._self_attention_stack(
                inputs,
                memory,
                decoder_self_attention_bias=decoder_self_attention_bias,
                memory_attention_bias=memory_attention_bias,
                cache=None,
                mode=mode)
            logits = self._output_layer(decoder_output)
            preds = tf.to_int32(tf.argmax(logits, axis=-1))
            rets = TransformerDecoderOutput(
                logits=logits,
                sample_id=preds
            )

        else:
            if max_decoding_length is None:
                max_decoding_length = self._hparams.max_decoding_length
            self.max_decoding_length = max_decoding_length
            if beam_width is None:  # Inference-like decoding
                # Prepare helper
                if helper is None:
                    if decoding_strategy == "infer_greedy":
                        helper = tx_helper.GreedyEmbeddingHelper(
                            embedding, start_tokens, end_token)
                    elif decoding_strategy == "infer_sample":
                        helper = tx_helper.SampleEmbeddingHelper(
                            embedding, start_tokens, end_token,
                            softmax_temperature)
                    else:
                        raise ValueError(
                            "Unknown decoding strategy: {}".format(
                                decoding_strategy))
                self._helper = helper

                self._cache = self._init_cache(memory, memory_attention_bias,
                                               beam_search_decoding=False)
                if context is not None:
                    self.context = tf.pad(
                        self.context,
                        [[0, 0],
                         [0, max_decoding_length - shape_list(self.context)[1]]]
                    )

                outputs, _, sequence_lengths = dynamic_decode(
                    decoder=self,
                    impute_finished=impute_finished,
                    maximum_iterations=max_decoding_length,
                    output_time_major=False,
                    scope=self.variable_scope)

                if context is not None:
                    # Here the length of sample_id will be larger than that
                    # of logit by 1, because there will be a additional
                    # start_token in the returned sample_id.
                    # the start_id should be the first token of the
                    # given context
                    outputs = TransformerDecoderOutput(
                        logits=outputs.logits,
                        sample_id=tf.concat(
                            [tf.expand_dims(start_tokens, 1),
                             outputs.sample_id],
                            axis=1
                        )
                    )
                    sequence_lengths = sequence_lengths + 1
                rets = outputs, sequence_lengths

            else:  # Beam-search decoding
                # Ignore `decoding_strategy`; Assume `helper` is not set
                if helper is not None:
                    raise ValueError("Must not set 'beam_width' and 'helper' "
                                     "simultaneously.")
                _batch_size = shape_list(start_tokens)[0]
                self._cache = self._init_cache(memory, memory_attention_bias,
                                               beam_search_decoding=True,
                                               batch_size=_batch_size)

                # The output format is different when running beam search
                sample_id, log_prob = self._beam_decode(
                    start_tokens,
                    end_token,
                    beam_width=beam_width,
                    length_penalty=length_penalty,
                    decode_length=max_decoding_length,
                )
                rets = {
                    'sample_id': sample_id,
                    'log_prob': log_prob
                }

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return rets

    def _self_attention_stack(self,
                              inputs,
                              memory,
                              decoder_self_attention_bias=None,
                              memory_attention_bias=None,
                              cache=None,
                              mode=None):
        """Stacked multihead attention module.
        """
        def _layer_norm(x, scope):
            return layers.layer_normalize(x, reuse=tf.AUTO_REUSE, scope=scope)

        inputs = tf.layers.dropout(inputs,
                                   rate=self._hparams.embedding_dropout,
                                   training=is_train_mode(mode))
        if cache is not None:
            if memory is not None:
                memory_attention_bias = \
                    cache['memory_attention_bias']
        else:
            assert decoder_self_attention_bias is not None

        # self.adj_masks is set at the beginning of _build()
        adj_masks = self.adj_masks

        x = inputs
        for i in range(self._hparams.num_blocks):
            layer_name = 'layer_{}'.format(i)
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name) as layer_scope:
                with tf.variable_scope("self_attention"):
                    graph_multihead_attention = \
                        self.multihead_attentions['self_att'][i]
                    selfatt_output = graph_multihead_attention(
                        queries=_layer_norm(x, layer_scope),
                        memory=None,
                        adj_masks=adj_masks,
                        memory_attention_bias=decoder_self_attention_bias,
                        cache=layer_cache,
                        mode=mode,
                    )
                    x = x + tf.layers.dropout(
                        selfatt_output,
                        rate=self._hparams.residual_dropout,
                        training=is_train_mode(mode),
                    )
                if memory is not None:
                    with tf.variable_scope('encdec_attention') as \
                            encdec_attention_scope:
                        graph_multihead_attention = \
                            self.multihead_attentions['encdec_att'][i]
                        encdec_output = graph_multihead_attention(
                            queries=_layer_norm(x, encdec_attention_scope),
                            memory=memory,
                            adj_masks=adj_masks,
                            memory_attention_bias=memory_attention_bias,
                            mode=mode,
                        )
                        x = x + tf.layers.dropout(
                            encdec_output,
                            rate=self._hparams.residual_dropout,
                            training=is_train_mode(mode))
                poswise_network = self.poswise_networks[i]
                with tf.variable_scope('past_poswise_ln') as \
                        past_poswise_ln_scope:
                    sub_output = tf.layers.dropout(
                        poswise_network(_layer_norm(x, past_poswise_ln_scope)),
                        rate=self._hparams.residual_dropout,
                        training=is_train_mode(mode),
                    )
                    x = x + sub_output

        return _layer_norm(x, scope=self.variable_scope)


