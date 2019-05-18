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
Various RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, too-many-locals

import collections
import copy

import tensorflow as tf
from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq import dynamic_decode
from tensorflow.contrib.seq2seq import tile_batch

from texar.modules.decoders.rnn_decoder_base import RNNDecoderBase
from texar.utils import utils

from texar.core import layers
from texar.utils.mode import is_train_mode, is_train_mode_py
from texar.module_base import ModuleBase
from texar.modules.decoders import rnn_decoder_helpers
from texar.utils.dtypes import is_callable
from texar.utils.shapes import shape_list
from texar.modules.decoders import tf_helpers as tx_helper

from models.dynamic_attention_wrapper import DynamicBahdanauAttention
from models.dynamic_attention_wrapper import DynamicAttentionWrapper

__all__ = [
    "AttentionRNNDecoderOutput",
    "DynamicAttentionRNNDecoder"
]

class AttentionRNNDecoderOutput(
        collections.namedtuple(
            "AttentionRNNDecoderOutput",
            ["logits", "sample_id", "cell_output",
             "attention_scores", "attention_context"])):
    """The outputs of attention RNN decoders that additionally include
    attention results.

    Attributes:
        logits: The outputs of RNN (at each step/of all steps) by applying the
            output layer on cell outputs. E.g., in
            :class:`~texar.modules.AttentionRNNDecoder`, this is a Tensor of
            shape `[batch_size, max_time, vocab_size]` after decoding.
        sample_id: The sampled results (at each step/of all steps). E.g., in
            :class:`~texar.modules.AttentionRNNDecoder` with decoding strategy
            of train_greedy, this
            is a Tensor of shape `[batch_size, max_time]` containing the
            sampled token indexes of all steps.
        cell_output: The output of RNN cell (at each step/of all steps).
            This is the results prior to the output layer. E.g., in
            AttentionRNNDecoder with default
            hyperparameters, this is a Tensor of
            shape `[batch_size, max_time, cell_output_size]` after decoding
            the whole sequence.
        attention_scores: A single or tuple of `Tensor`(s) containing the
            alignments emitted (at the previous time step/of all time steps)
            for each attention mechanism.
        attention_context: The attention emitted (at the previous time step/of
            all time steps).
    """
    pass

class DynamicAttentionRNNDecoder(RNNDecoderBase):
    """RNN decoder with attention mechanism.

    Args:
        memory: The memory to query, e.g., the output of an RNN encoder. This
            tensor should be shaped `[batch_size, max_time, dim]`.
        memory_sequence_length (optional): A tensor of shape `[batch_size]`
            containing the sequence lengths for the batch
            entries in memory. If provided, the memory tensor rows are masked
            with zeros for values past the respective sequence lengths.
        cell (RNNCell, optional): An instance of `RNNCell`. If `None`, a cell
            is created as specified in :attr:`hparams`.
        cell_dropout_mode (optional): A Tensor taking value of
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, which
            toggles dropout in the RNN cell (e.g., activates dropout in
            TRAIN mode). If `None`, :func:`~texar.global_mode` is used.
            Ignored if :attr:`cell` is given.
        vocab_size (int, optional): Vocabulary size. Required if
            :attr:`output_layer` is `None`.
        output_layer (optional): An output layer that transforms cell output
            to logits. This can be:

            - A callable layer, e.g., an instance \
            of :tf_main:`tf.layers.Layer <layers/Layer>`.
            - A tensor. A dense layer will be created using the tensor \
            as the kernel weights. The bias of the dense layer is determined by\
            `hparams.output_layer_bias`. This can be used to tie the output \
            layer with the input embedding matrix, as proposed in \
            https://arxiv.org/pdf/1608.05859.pdf
            - `None`. A dense layer will be created based on attr:`vocab_size`\
            and `hparams.output_layer_bias`.
            - If no output layer after the cell output is needed, set \
            `(vocab_size=None, output_layer=tf.identity)`.
        cell_input_fn (callable, optional): A callable that produces RNN cell
            inputs. If `None` (default), the default is used:
            `lambda inputs, attention: tf.concat([inputs, attention], -1)`,
            which cancats regular RNN cell inputs with attentions.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    See :meth:`~texar.modules.RNNDecoderBase._build` for the inputs and outputs
    of the decoder. The decoder returns
    `(outputs, final_state, sequence_lengths)`, where `outputs` is an instance
    of :class:`~texar.modules.AttentionRNNDecoderOutput`.

    Example:

        .. code-block:: python

            # Encodes the source
            enc_embedder = WordEmbedder(data.source_vocab.size, ...)
            encoder = UnidirectionalRNNEncoder(...)

            enc_outputs, _ = encoder(
                inputs=enc_embedder(data_batch['source_text_ids']),
                sequence_length=data_batch['source_length'])

            # Decodes while attending to the source
            dec_embedder = WordEmbedder(vocab_size=data.target_vocab.size, ...)
            decoder = AttentionRNNDecoder(
                memory=enc_outputs,
                memory_sequence_length=data_batch['source_length'],
                vocab_size=data.target_vocab.size)

            outputs, _, _ = decoder(
                decoding_strategy='train_greedy',
                inputs=dec_embedder(data_batch['target_text_ids']),
                sequence_length=data_batch['target_length']-1)
    """
    def __init__(self,
                 memory_sequence_length=None,
                 cell=None,
                 cell_dropout_mode=None,
                 vocab_size=None,
                 output_layer=None,
                 #attention_layer=None, # TODO(zhiting): only valid for tf>=1.0
                 cell_input_fn=None,
                 hparams=None):
        RNNDecoderBase.__init__(
            self, cell, vocab_size, output_layer, cell_dropout_mode, hparams)

        attn_hparams = self._hparams['attention']
        attn_kwargs = attn_hparams['kwargs'].todict()

        # Parse the 'probability_fn' argument
        if 'probability_fn' in attn_kwargs:
            prob_fn = attn_kwargs['probability_fn']
            if prob_fn is not None and not callable(prob_fn):
                prob_fn = utils.get_function(
                    prob_fn,
                    ['tensorflow.nn', 'tensorflow.contrib.sparsemax',
                     'tensorflow.contrib.seq2seq'])
            attn_kwargs['probability_fn'] = prob_fn

        
        # attn_kwargs.update({
        #     "memory_sequence_length": memory_sequence_length,
        #     "memory": memory})
        attn_kwargs.update({
            "memory_sequence_length": memory_sequence_length
            })
        self._attn_kwargs = attn_kwargs
        attn_modules = ['tensorflow.contrib.seq2seq', 'texar.custom', 'models.dynamic_attention_wrapper']
        # Use variable_scope to ensure all trainable variables created in
        # the attention mechanism are collected
        with tf.variable_scope(self.variable_scope):
            attention_mechanism = utils.check_or_get_instance(
                attn_hparams["type"], attn_kwargs, attn_modules,
                classtype=tf.contrib.seq2seq.AttentionMechanism)

        self._attn_cell_kwargs = {
            "attention_layer_size": attn_hparams["attention_layer_size"],
            "alignment_history": attn_hparams["alignment_history"],
            "output_attention": attn_hparams["output_attention"],
        }
        self._cell_input_fn = cell_input_fn
        # Use variable_scope to ensure all trainable variables created in
        # AttentionWrapper are collected
        with tf.variable_scope(self.variable_scope):
            #if attention_layer is not None:
            #    self._attn_cell_kwargs["attention_layer_size"] = None
            attn_cell = AttentionWrapper(
                self._cell,
                attention_mechanism, # DynamicBahaudauAttention
                cell_input_fn=self._cell_input_fn,
                #attention_layer=attention_layer,
                **self._attn_cell_kwargs)
            self._cell = attn_cell
    
    # heritated from RNNDecoderBase
    def _build(self,
               decoding_strategy="train_greedy",
               initial_state=None,
               inputs=None,
               memory=None,
               sequence_length=None,
               embedding=None,
               start_tokens=None,
               end_token=None,
               softmax_temperature=None,
               max_decoding_length=None,
               impute_finished=False,
               output_time_major=False,
               input_time_major=False,
               helper=None,
               mode=None,
               **kwargs):
        # Memory
        for _mechanism in self._cell._attention_mechanisms:
            _mechanism.initialize_memory(memory)
        # Helper
        if helper is not None:
            pass
        elif decoding_strategy is not None:
            if decoding_strategy == "train_greedy":
                helper = rnn_decoder_helpers._get_training_helper(
                    inputs, sequence_length, embedding, input_time_major)
            elif decoding_strategy == "infer_greedy":
                helper = tx_helper.GreedyEmbeddingHelper(
                    embedding, start_tokens, end_token)
            elif decoding_strategy == "infer_sample":
                helper = tx_helper.SampleEmbeddingHelper(
                    embedding, start_tokens, end_token, softmax_temperature)
            else:
                raise ValueError(
                    "Unknown decoding strategy: {}".format(decoding_strategy))
        else:
            if is_train_mode_py(mode):
                kwargs_ = copy.copy(self._hparams.helper_train.kwargs.todict())
                helper_type = self._hparams.helper_train.type
            else:
                kwargs_ = copy.copy(self._hparams.helper_infer.kwargs.todict())
                helper_type = self._hparams.helper_infer.type
            kwargs_.update({
                "inputs": inputs,
                "sequence_length": sequence_length,
                "time_major": input_time_major,
                "embedding": embedding,
                "start_tokens": start_tokens,
                "end_token": end_token,
                "softmax_temperature": softmax_temperature})
            kwargs_.update(kwargs)
            helper = rnn_decoder_helpers.get_helper(helper_type, **kwargs_)
        self._helper = helper

        # Initial state
        if initial_state is not None:
            self._initial_state = initial_state
        else:
            self._initial_state = self.zero_state(
                batch_size=self.batch_size, dtype=tf.float32)

        # Maximum decoding length
        max_l = max_decoding_length
        if max_l is None:
            max_l_train = self._hparams.max_decoding_length_train
            if max_l_train is None:
                max_l_train = utils.MAX_SEQ_LENGTH
            max_l_infer = self._hparams.max_decoding_length_infer
            if max_l_infer is None:
                max_l_infer = utils.MAX_SEQ_LENGTH
            max_l = tf.cond(is_train_mode(mode),
                            lambda: max_l_train, lambda: max_l_infer)
        self.max_decoding_length = max_l
        # Decode
        outputs, final_state, sequence_lengths = dynamic_decode(
            decoder=self, impute_finished=impute_finished,
            maximum_iterations=max_l, output_time_major=output_time_major)

        if not self._built:
            self._add_internal_trainable_variables()
            # Add trainable variables of `self._cell` which may be
            # constructed externally.
            self._add_trainable_variable(
                layers.get_rnn_cell_trainable_variables(self._cell))
            if isinstance(self._output_layer, tf.layers.Layer):
                self._add_trainable_variable(
                    self._output_layer.trainable_variables)
            # Add trainable variables of `self._beam_search_rnn_cell` which
            # may already be constructed and used.
            if self._beam_search_cell is not None:
                self._add_trainable_variable(
                    self._beam_search_cell.trainable_variables)

            self._built = True

        return outputs, final_state, sequence_lengths

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values:

        Common hyperparameters are the same as in
        :class:`~texar.modules.BasicRNNDecoder`.
        :meth:`~texar.modules.BasicRNNDecoder.default_hparams`.
        Additional hyperparameters are for attention mechanism
        configuration.

        .. code-block:: python

            {
                "attention": {
                    "type": "LuongAttention",
                    "kwargs": {
                        "num_units": 256,
                    },
                    "attention_layer_size": None,
                    "alignment_history": False,
                    "output_attention": True,
                },
                # The following hyperparameters are the same as with
                # `BasicRNNDecoder`
                "rnn_cell": default_rnn_cell_hparams(),
                "max_decoding_length_train": None,
                "max_decoding_length_infer": None,
                "helper_train": {
                    "type": "TrainingHelper",
                    "kwargs": {}
                }
                "helper_infer": {
                    "type": "SampleEmbeddingHelper",
                    "kwargs": {}
                }
                "name": "attention_rnn_decoder"
            }

        Here:

        "attention" : dict
            Attention hyperparameters, including:

            "type" : str or class or instance
                The attention type. Can be an attention class, its name or
                module path, or a class instance. The class must be a subclass
                of :tf_main:`TF AttentionMechanism
                <contrib/seq2seq/AttentionMechanism>`. If class name is
                given, the class must be from modules
                :tf_main:`tf.contrib.seq2seq <contrib/seq2seq>` or
                :mod:`texar.custom`.

                Example:

                    .. code-block:: python

                        # class name
                        "type": "LuongAttention"
                        "type": "BahdanauAttention"
                        # module path
                        "type": "tf.contrib.seq2seq.BahdanauMonotonicAttention"
                        "type": "my_module.MyAttentionMechanismClass"
                        # class
                        "type": tf.contrib.seq2seq.LuongMonotonicAttention
                        # instance
                        "type": LuongAttention(...)

            "kwargs" : dict
                keyword arguments for the attention class constructor.
                Arguments :attr:`memory` and
                :attr:`memory_sequence_length` should **not** be
                specified here because they are given to the decoder
                constructor. Ignored if "type" is an attention class
                instance. For example

                Example:

                    .. code-block:: python

                        "type": "LuongAttention",
                        "kwargs": {
                            "num_units": 256,
                            "probability_fn": tf.nn.softmax
                        }

                    Here "probability_fn" can also be set to the string name
                    or module path to a probability function.

                "attention_layer_size" : int or None
                    The depth of the attention (output) layer. The context and
                    cell output are fed into the attention layer to generate
                    attention at each time step.
                    If `None` (default), use the context as attention at each
                    time step.

                "alignment_history": bool
                    whether to store alignment history from all time steps
                    in the final output state. (Stored as a time major
                    `TensorArray` on which you must call `stack()`.)

                "output_attention": bool
                    If `True` (default), the output at each time step is
                    the attention value. This is the behavior of Luong-style
                    attention mechanisms. If `False`, the output at each
                    time step is the output of `cell`.  This is the
                    beahvior of Bhadanau-style attention mechanisms.
                    In both cases, the `attention` tensor is propagated to
                    the next time step via the state and is used there.
                    This flag only controls whether the attention mechanism
                    is propagated up to the next cell in an RNN stack or to
                    the top RNN output.
        """
        hparams = RNNDecoderBase.default_hparams()
        hparams["name"] = "attention_rnn_decoder"
        hparams["attention"] = {
            "type": "LuongAttention",
            "kwargs": {
                "num_units": 256,
            },
            "attention_layer_size": None,
            "alignment_history": False,
            "output_attention": True,
        }
        return hparams

    # pylint: disable=arguments-differ
    def _get_beam_search_cell(self, beam_width):
        """Returns the RNN cell for beam search decoding.
        """
        with tf.variable_scope(self.variable_scope, reuse=True):
            attn_kwargs = copy.copy(self._attn_kwargs)

            memory = attn_kwargs['memory']
            attn_kwargs['memory'] = tile_batch(memory, multiplier=beam_width)

            memory_seq_length = attn_kwargs['memory_sequence_length']
            if memory_seq_length is not None:
                attn_kwargs['memory_sequence_length'] = tile_batch(
                    memory_seq_length, beam_width)

            attn_modules = ['tensorflow.contrib.seq2seq', 'texar.custom']
            bs_attention_mechanism = utils.check_or_get_instance(
                self._hparams.attention.type, attn_kwargs, attn_modules,
                classtype=tf.contrib.seq2seq.AttentionMechanism)

            bs_attn_cell = AttentionWrapper(
                self._cell._cell,
                bs_attention_mechanism,
                cell_input_fn=self._cell_input_fn,
                **self._attn_cell_kwargs)

            self._beam_search_cell = bs_attn_cell

            return bs_attn_cell

    def initialize(self, name=None):
        helper_init = self._helper.initialize()

        flat_initial_state = nest.flatten(self._initial_state)
        dtype = flat_initial_state[0].dtype
        initial_state = self._cell.zero_state(
            batch_size=tf.shape(flat_initial_state[0])[0], dtype=dtype)
        initial_state = initial_state.clone(cell_state=self._initial_state)

        return [helper_init[0], helper_init[1], initial_state]

    def step(self, time, inputs, state, name=None):
        wrapper_outputs, wrapper_state = self._cell(inputs, state)
        # Essentisally the same as in BasicRNNDecoder.step()
        logits = self._output_layer(wrapper_outputs)
        sample_ids = self._helper.sample(
            time=time, outputs=logits, state=wrapper_state)
        reach_max_time = tf.equal(time+1, self.max_decoding_length)

        (finished, next_inputs, next_state) = self._helper.next_inputs(
            time=time,
            outputs=logits,
            state=wrapper_state,
            sample_ids=sample_ids,
            reach_max_time=reach_max_time)

        attention_scores = wrapper_state.alignments
        attention_context = wrapper_state.attention
        outputs = AttentionRNNDecoderOutput(
            logits, sample_ids, wrapper_outputs,
            attention_scores, attention_context)

        return (outputs, next_state, next_inputs, finished)

    def finalize(self, outputs, final_state, sequence_lengths):
        return outputs, final_state

    def _alignments_size(self):
        # Reimplementation of the alignments_size of each of
        # AttentionWrapper.attention_mechanisms. The original implementation
        # of `_BaseAttentionMechanism._alignments_size`:
        #
        #    self._alignments_size = (self._keys.shape[1].value or
        #                       array_ops.shape(self._keys)[1])
        #
        # can be `None` when the seq length of encoder outputs are priori
        # unknown.
        alignments_size = []
        for am in self._cell._attention_mechanisms:
            az = (am._keys.shape[1].value or tf.shape(am._keys)[1:-1])
            alignments_size.append(az)
        return self._cell._item_or_tuple(alignments_size)

    @property
    def output_size(self):
        return AttentionRNNDecoderOutput(
            logits=self._rnn_output_size(),
            sample_id=self._helper.sample_ids_shape,
            cell_output=self._cell.output_size,
            attention_scores=self._alignments_size(),
            attention_context=self._cell.state_size.attention)

    @property
    def output_dtype(self):
        """Types of output of one step.
        """
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = nest.flatten(self._initial_state)[0].dtype
        return AttentionRNNDecoderOutput(
            logits=nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            sample_id=self._helper.sample_ids_dtype,
            cell_output=nest.map_structure(
                lambda _: dtype, self._cell.output_size),
            attention_scores=nest.map_structure(
                lambda _: dtype, self._alignments_size()),
            attention_context=nest.map_structure(
                lambda _: dtype, self._cell.state_size.attention))

    def zero_state(self, batch_size, dtype):
        """Returns zero state of the basic cell.
        Equivalent to :attr:`decoder.cell._cell.zero_state`.
        """
        return self._cell._cell.zero_state(batch_size=batch_size, dtype=dtype)

    def wrapper_zero_state(self, batch_size, dtype):
        """Returns zero state of the attention-wrapped cell.
        Equivalent to :attr:`decoder.cell.zero_state`.
        """
        return self._cell.zero_state(batch_size=batch_size, dtype=dtype)

    @property
    def state_size(self):
        """The state size of the basic cell.
        Equivalent to :attr:`decoder.cell._cell.state_size`.
        """
        return self._cell._cell.state_size

    @property
    def wrapper_state_size(self):
        """The state size of the attention-wrapped cell.
        Equivalent to :attr:`decoder.cell.state_size`.
        """
        return self._cell.state_size
