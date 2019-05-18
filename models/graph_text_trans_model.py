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
"""Text style transfer
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, too-many-locals

import tensorflow as tf

import texar as tx
from texar.modules import WordEmbedder
from texar.modules import UnidirectionalRNNEncoder
from texar.modules import MLPTransformConnector
from texar.modules import AttentionRNNDecoder
from texar.modules import BasicRNNDecoder
from texar.modules import GumbelSoftmaxEmbeddingHelper
from texar.modules import Conv1DClassifier

from texar.core import get_train_op
from texar.utils import collect_trainable_variables, get_batch_size

from texar.modules import TransformerEncoder
from texar.utils import transformer_utils

from models.self_graph_transformer import SelfGraphTransformerEncoder
from models.cross_graph_transformer import CrossGraphTransformerFixedLengthDecoder

from models.rnn_dynamic_decoders import DynamicAttentionRNNDecoder


class GraphTextTransModel(object):
    """Control  
    """

    def __init__(self, inputs, vocab, gamma, lambda_g_graph, lambda_g_sentence, hparams=None):
        self._hparams = tx.HParams(hparams, None)
        self._build_model(inputs, vocab, gamma, lambda_g_graph, lambda_g_sentence)

    def _build_model(self, inputs, vocab, gamma, lambda_g_graph, lambda_g_sentence):
        """Builds the model.
        """
        embedder = WordEmbedder(
            vocab_size=vocab.size,
            hparams=self._hparams.embedder
        )
        encoder = SelfGraphTransformerEncoder(
            hparams=self._hparams.encoder
        )

        # text_ids for encoder, with the first token being the BOS token
        enc_text_ids = inputs['text_ids']
        sequence_length = inputs['length']

        # pre_embedding_text_ids: [batch, max_time-1, dim]
        pre_embedding_text_ids = embedder(enc_text_ids)[:, 1:, :]

        # change the BOS token embedding to be label embedding
        label_connector = MLPTransformConnector(self._hparams.dim_c)
        labels = tf.to_float(tf.reshape(inputs['labels'], [-1, 1]))
        # c and c_: [batch_size, 1, dim]
        c = tf.expand_dims(label_connector(labels), 1)
        c_ = tf.expand_dims(label_connector(1 - labels), 1)

        # embedding_text_ids and embedding_text_ids_: [batch_size, max_time, dim]
        # embedding_text_ids: token embeddings with the original style embedded in the first BOS token
        # embedding_text_ids_: token embeddings with the transfered style embedded in the first BOS token
        embedding_text_ids = tf.concat([c, pre_embedding_text_ids], axis=1)
        embedding_text_ids_ = tf.concat([c_, pre_embedding_text_ids], axis=1)

        # adjs need to be corrected for input graph structures
        # enc_shape = tf.shape(embedding_text_ids)
        # adjs = tf.ones([enc_shape[0], enc_shape[1], enc_shape[1]])
        enc_shape = tf.shape(embedding_text_ids)
        adjs = tf.to_int32(tf.reshape(inputs['adjs'], [-1,17,17]))
        # adjs = inputs['adjs']
        print("adjs shape:",tf.shape(adjs))
        adjs = adjs[:, :enc_shape[1], :enc_shape[1]]
        print("adjs reshape:", tf.shape(adjs))    #Tensor("Shape_2:0", shape=(3,), dtype=int32)

        enc_outputs = encoder(
            inputs = embedding_text_ids, 
            sequence_length = sequence_length, 
            adjs = adjs
        )
        enc_outputs_ = encoder(
            inputs = embedding_text_ids_, 
            sequence_length = sequence_length, 
            adjs = adjs
        )

        # Creates classfier for graph
        classifier_graph = Conv1DClassifier(hparams=self._hparams.classifier)

        # Classification loss for classifier_graph when keeping original style
        clas_logits_graph, clas_preds_graph = classifier_graph(
            inputs = embedding_text_ids[:, 1:, :],
            sequence_length = sequence_length - 1
        )
        loss_d_clas_graph = tf.nn.sigmoid_cross_entropy_with_logits(
            labels = tf.to_float(inputs['labels']),
            logits = clas_logits_graph
        )
        loss_d_clas_graph = tf.reduce_mean(loss_d_clas_graph)
        accu_d_graph = tx.evals.accuracy(
            labels = inputs['labels'], 
            preds=clas_preds_graph
        )

        # Classification loss for SelfGraphTransformer and classifier_graph when transferring style
        trans_logits_graph, trans_preds_graph = classifier_graph(
            inputs = enc_outputs_[:, 1:, :],
            sequence_length = sequence_length - 1
        )
        loss_g_clas_graph = tf.nn.sigmoid_cross_entropy_with_logits(
            labels = tf.to_float(1 - inputs['labels']),
            logits = trans_logits_graph
        )
        loss_g_clas_graph = tf.reduce_mean(loss_g_clas_graph)
        accu_g_graph = tx.evals.accuracy(
            labels = 1 - inputs['labels'],
            preds = trans_preds_graph
        )

        # decoder is a CrossGraphTransformerFixedLengthDecoder but with encoder_output for rephrase
        # decoder shares the same hparams with encoder
        # rephrase_encoder and rephrase_decoder are used to rewrite a natural sentence
        decoder = CrossGraphTransformerFixedLengthDecoder(
            vocab_size = vocab.size,
            tau = gamma,
            hparams = self._hparams.encoder
        )
        rephrase_encoder = UnidirectionalRNNEncoder(hparams=self._hparams.rephrase_encoder)
        rephrase_decoder =DynamicAttentionRNNDecoder(
            memory_sequence_length = sequence_length - 1,
            cell_input_fn=lambda inputs, attention: inputs,
            vocab_size=vocab.size,
            hparams=self._hparams.rephrase_decoder)

        # Auto-encoding loss for G
        # The first token that represents BOS/CLS is removed
        # Currently use the same sequence_length and memory_sequence_length
        # Later we may consider use the CLS flag in embedding_text_ids to guide the generation
        g_outputs = decoder(
            inputs = enc_outputs[:, 1:, :], 
            memory = pre_embedding_text_ids,
            sequence_length = sequence_length - 1, 
            memory_sequence_length = sequence_length-1,
            adjs = adjs[:, 1:, 1:],
            encoder_output = True
        )
        rephrase_enc, rephrase_state = rephrase_encoder(
            g_outputs, 
            sequence_length = sequence_length - 1
        )
        rephrase_outputs, _, _ = rephrase_decoder(
            initial_state = rephrase_state,
            memory = rephrase_enc, # embedder(inputs['text_ids'][:, 1:]),
            sequence_length = sequence_length - 1,
            inputs = inputs['text_ids'],
            embedding = embedder
        )
        loss_g_ae = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels = inputs['text_ids'][:, 1:],
            logits = rephrase_outputs.logits,
            sequence_length = sequence_length - 1,
            average_across_timesteps = True,
            sum_over_timesteps = False
        )

        # Creates classifier for sentence
        classifier_sentence = Conv1DClassifier(hparams=self._hparams.classifier)
        clas_embedder = WordEmbedder(
            vocab_size = vocab.size,
            hparams = self._hparams.embedder
        )

        # Classification loss for the classifier
        clas_logits_sentence, clas_preds_sentence = classifier_sentence(
            inputs = clas_embedder(ids = inputs['text_ids'][:, 1:]),
            sequence_length = sequence_length - 1
        )
        loss_d_clas_sentence = tf.nn.sigmoid_cross_entropy_with_logits(
            labels = tf.to_float(inputs['labels']), 
            logits = clas_logits_sentence
        )
        loss_d_clas_sentence = tf.reduce_mean(loss_d_clas_sentence)
        accu_d_sentence = tx.evals.accuracy(
            labels = inputs['labels'], 
            preds = clas_preds_sentence
        )

        # Classification loss for the generator, based on soft samples
        # Continuous softmax decoding, used in training
        # We will consider Gumbel-softmax decoding
        g_outputs_ = decoder(
            inputs = enc_outputs_[:, 1:, :],
            memory = pre_embedding_text_ids,
            sequence_length = sequence_length - 1,
            memory_sequence_length = sequence_length - 1,
            adjs = adjs[:, 1:, 1:],
            encoder_output = True
        )

        # Gumbel-softmax decoding, used in training
        start_tokens = tf.ones_like(inputs['labels']) * vocab.bos_token_id
        end_token = vocab.eos_token_id
        gumbel_helper = GumbelSoftmaxEmbeddingHelper(
            embedder.embedding, start_tokens, end_token, gamma)

        rephrase_enc_, rephrase_state_ = rephrase_encoder(
            g_outputs_, 
            sequence_length = sequence_length - 1
        )
        soft_rephrase_outputs_, _, soft_rephrase_length_ = rephrase_decoder(
            memory = rephrase_enc_, # embedder(inputs['text_ids'][:, 1:]),
            helper = gumbel_helper,
            initial_state = rephrase_state_
        )

        # Greedy decoding, used in eval
        rephrase_outputs_, _, rephrase_length_ = rephrase_decoder(
            decoding_strategy = 'infer_greedy',
            memory = rephrase_enc_,
            initial_state = rephrase_state_,
            embedding = embedder,
            start_tokens = start_tokens,
            end_token = end_token
        )
        soft_logits_sentence, soft_preds_sentence = classifier_sentence(
            inputs = clas_embedder(soft_ids=soft_rephrase_outputs_.sample_id),
            sequence_length = soft_rephrase_length_
        )
        loss_g_clas_sentence = tf.nn.sigmoid_cross_entropy_with_logits(
            labels = tf.to_float(1 - inputs['labels']), 
            logits = soft_logits_sentence
        )
        loss_g_clas_sentence = tf.reduce_mean(loss_g_clas_sentence)

        # Accuracy on soft samples, for training progress monitoring
        accu_g_sentence = tx.evals.accuracy(
            labels = 1 - inputs['labels'], 
            preds = soft_preds_sentence
        )

        # Accuracy on greedy-decoded samples, for training progress monitoring
        _, gdy_preds_sentence = classifier_sentence(
            inputs = clas_embedder(ids=rephrase_outputs_.sample_id),
            sequence_length = rephrase_length_
        )
        accu_g_gdy_sentence = tx.evals.accuracy(
            labels = 1 - inputs['labels'],
            preds = gdy_preds_sentence
        )

        # Aggregates losses
        loss_g = loss_g_ae + lambda_g_graph * loss_g_clas_graph + lambda_g_sentence * loss_g_clas_sentence
        loss_d = loss_d_clas_graph + loss_d_clas_sentence

        # Creates optimizers
        g_vars = collect_trainable_variables(
            [embedder, encoder, label_connector, decoder, rephrase_encoder, rephrase_decoder])
        d_vars = collect_trainable_variables([clas_embedder, classifier_graph, classifier_sentence])

        train_op_g = get_train_op(
            loss_g, g_vars, hparams=self._hparams.opt)
        train_op_g_ae = get_train_op(
            loss_g_ae, g_vars, hparams=self._hparams.opt)
        train_op_d = get_train_op(
            loss_d, d_vars, hparams=self._hparams.opt)

        # Interface tensors
        self.losses = {
            "loss_g": loss_g,
            "loss_d": loss_d,
            "loss_g_ae": loss_g_ae,
            "loss_g_clas_graph": loss_g_clas_graph,
            "loss_g_clas_sentence": loss_g_clas_sentence,
            "loss_d_clas_graph": loss_d_clas_graph,
            "loss_d_clas_sentence": loss_d_clas_sentence,
        }
        self.metrics = {
            "accu_d_graph": accu_d_graph,
            "accu_d_sentence": accu_d_sentence,
            "accu_g_graph": accu_g_graph,
            "accu_g_sentence": accu_g_sentence,
            "accu_g_gdy_sentence": accu_g_gdy_sentence
        }
        self.train_ops = {
            "train_op_g": train_op_g,
            "train_op_g_ae": train_op_g_ae,
            "train_op_d": train_op_d
        }
        self.samples = {
            "original": inputs['text_ids'][:, 1:],
            "transferred": rephrase_outputs_.sample_id
        }

        self.fetches_train_g = {
            "loss_g": self.train_ops["train_op_g"],
            "loss_g_ae": self.losses["loss_g_ae"],
            "loss_g_clas_graph": self.losses["loss_g_clas_graph"],
            "loss_g_clas_sentence": self.losses["loss_g_clas_sentence"],
            "accu_g_graph": self.metrics["accu_g_graph"],
            "accu_g_sentence": self.metrics["accu_g_sentence"],
            "accu_g_gdy_sentence": self.metrics["accu_g_gdy_sentence"]
        }
        self.fetches_train_d = {
            "loss_d": self.train_ops["train_op_d"],
            "loss_d_clas_graph": self.losses["loss_d_clas_graph"],
            "loss_d_clas_sentence": self.losses["loss_d_clas_sentence"],
            "accu_d_graph": self.metrics["accu_d_graph"],
            "accu_d_sentence": self.metrics["accu_d_sentence"]
        }
        fetches_eval = {"batch_size": get_batch_size(inputs['text_ids'])}
        fetches_eval.update(self.losses)
        fetches_eval.update(self.metrics)
        fetches_eval.update(self.samples)
        self.fetches_eval = fetches_eval

