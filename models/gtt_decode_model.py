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
from texar.modules import TransformerDecoder
from texar.modules import GumbelSoftmaxEmbeddingHelper
from texar.modules import Conv1DClassifier

from texar.core import get_train_op
from texar.utils import collect_trainable_variables, get_batch_size

from texar.modules import TransformerEncoder
from texar.utils import transformer_utils

from models.self_graph_transformer import SelfGraphTransformerEncoder
from models.cross_graph_transformer import CrossGraphTransformerSequentialDecoder

from models.rnn_dynamic_decoders import DynamicAttentionRNNDecoder


class GraphTextTransDecodeModel(object):
    """Control  
    """
    def __init__(self, inputs, vocab, gamma, lambda_g_graph, lambda_g_sentence, hparams=None):
        self._hparams = tx.HParams(hparams, None)
        self._prepare_inputs(inputs, vocab, gamma, lambda_g_graph, lambda_g_sentence),
        self._build_model()
    

    def _prepare_inputs(self, inputs, vocab, gamma, lambda_g_graph, lambda_g_sentence):
        self.vocab = vocab
        self.gamma = gamma
        self.lambda_g_graph = lambda_g_graph
        self.lambda_g_sentence = lambda_g_sentence

        # the first token is the BOS token
        self.text_ids = inputs['text_ids']
        self.sequence_length = inputs['length']
        self.labels = inputs['labels']

        enc_shape = tf.shape(self.text_ids)
        adjs = tf.to_int32(tf.reshape(inputs['adjs'], [-1,17,17]))
        self.adjs = adjs[:, :enc_shape[1], :enc_shape[1]]
        # self.adjs = tf.ones([enc_shape[0], enc_shape[1], enc_shape[1]])


    def _build_model(self):
        """Builds the model.
        """
        self._prepare_modules()
        self._build_self_graph_encoder()
        self._build_decoder()
        self._get_loss_train_op()
    
    
    def _prepare_modules(self):
        """Prepare necessary modules
        """
        self.embedder = WordEmbedder(vocab_size=self.vocab.size, hparams=self._hparams.embedder)
        self.clas_embedder = WordEmbedder(vocab_size=self.vocab.size, hparams=self._hparams.embedder)
        self.label_connector = MLPTransformConnector(self._hparams.dim_c)

        self.self_graph_encoder = SelfGraphTransformerEncoder(hparams=self._hparams.encoder)
        self.decoder = DynamicAttentionRNNDecoder(
            memory_sequence_length = self.sequence_length - 1,
            cell_input_fn = lambda inputs, attention: inputs,
            vocab_size = self.vocab.size,
            hparams = self._hparams.decoder
        )
        # TransformerDecoder will lead to OOM
        # self.decoder = TransformerDecoder(
        #     vocab_size = self.vocab.size,
        #     hparams = self._hparams.decoder
        # )
        # self.cross_graph_decoder = CrossGraphTransformerSequentialDecoder(
        #     vocab_size = self.vocab.size,
        #     hparams = self._hparams.decoder
        # )

        self.classifier_graph = Conv1DClassifier(hparams=self._hparams.classifier)
        self.classifier_sentence = Conv1DClassifier(hparams=self._hparams.classifier)


    def _build_self_graph_encoder(self):
        """
        Use: 
            self.embedder, self.self_graph_encoder
            self.text_ids, self.pre_embedding_text_ids, 

        Create:
            self.pre_embedding_text_ids,
            self.embedding_text_ids, self.embedding_text_ids_,
            self.enc_outputs, self.enc_outputs_

        """
        self.pre_embedding_text_ids = self.embedder(self.text_ids)[:, 1:, :]

        # change the BOS token embedding to be label embedding
        # c and c_: [batch_size, 1, dim]
        labels = tf.to_float(tf.reshape(self.labels, [-1, 1]))
        c = tf.expand_dims(self.label_connector(labels), 1)
        c_ = tf.expand_dims(self.label_connector(1 - labels), 1)

        # embedding_text_ids and embedding_text_ids_: [batch_size, max_time, dim]
        # embedding_text_ids: token embeddings with the original style embedded in the first BOS token
        # embedding_text_ids_: token embeddings with the transfered style embedded in the first BOS token
        self.embedding_text_ids = tf.concat([c, self.pre_embedding_text_ids], axis=1)
        self.embedding_text_ids_ = tf.concat([c_, self.pre_embedding_text_ids], axis=1)

        # encoding output for original style
        self.enc_outputs = self.self_graph_encoder(
            inputs = self.embedding_text_ids, 
            sequence_length = self.sequence_length, 
            adjs = self.adjs
        )
        # encoding output_ for transferred style
        self.enc_outputs_ = self.self_graph_encoder(
            inputs = self.embedding_text_ids_, 
            sequence_length = self.sequence_length, 
            adjs = self.adjs
        )

        # self.enc_outputs_state = self.initial_state_connector(self.enc_outputs)
        # self.enc_outputs_state_ = self.initial_state_connector(self.enc_outputs_)

        self.enc_outputs_state = tf.map_fn(fn=lambda x: tf.reduce_mean(x[0][1:x[1], :], axis=0), elems=(self.enc_outputs, self.sequence_length), dtype=tf.float32)
        self.enc_outputs_state_ = tf.map_fn(fn=lambda x: tf.reduce_mean(x[0][1:x[1], :], axis=0), elems=(self.enc_outputs_, self.sequence_length), dtype=tf.float32)

        self._train_ori_clas_graph()
        self._train_trans_clas_graph()
    

    def _train_ori_clas_graph(self):
        """Classification loss for classifier_graph when keeping original style
        Use: 
            self.classifier_graph

        Create:
            self.loss_d_clas_graph, self.accu_d_graph
        """
        # Do not consider the style node embedding
        clas_logits_graph, clas_preds_graph = self.classifier_graph(
            inputs = self.embedding_text_ids[:, 1:, :],
            sequence_length = self.sequence_length - 1
        )
        loss_d_clas_graph = tf.nn.sigmoid_cross_entropy_with_logits(
            labels = tf.to_float(self.labels),
            logits = clas_logits_graph
        )
        self.loss_d_clas_graph = tf.reduce_mean(loss_d_clas_graph)
        self.accu_d_graph = tx.evals.accuracy(
            labels = self.labels, 
            preds = clas_preds_graph
        )


    def _train_trans_clas_graph(self):
        """Classification loss for SelfGraphTransformer and classifier_graph when transferring style
        Use: 
            self.classifier_graph

        Create:
            self.loss_g_clas_graph, self.accu_g_graph
        """
        # Do not consider the style node embedding
        trans_logits_graph, trans_preds_graph = self.classifier_graph(
            inputs = self.enc_outputs_[:, 1:, :],
            sequence_length = self.sequence_length - 1
        )
        loss_g_clas_graph = tf.nn.sigmoid_cross_entropy_with_logits(
            labels = tf.to_float(1 - self.labels),
            logits = trans_logits_graph
        )
        self.loss_g_clas_graph = tf.reduce_mean(loss_g_clas_graph)
        self.accu_g_graph = tx.evals.accuracy(
            labels = 1 - self.labels,
            preds = trans_preds_graph
        )

    
    def _build_decoder(self):
        """
        Use: 
            self.decoder, self.enc_outputs
            self.pre_embedding_text_ids

        Create:
            self.g_outputs, self.g_outputs_
        """
        # Auto-encoding loss for G
        # The first token that represents BOS/CLS is removed
        # Currently use the same sequence_length and memory_sequence_length
        # Later we may consider use the CLS flag in embedding_text_ids to guide the generation
        self.g_outputs, _, _ = self.decoder(
            initial_state = self.enc_outputs_state,
            memory = self.enc_outputs[:, 1:, :],
            sequence_length = self.sequence_length - 1,
            inputs = self.text_ids,
            embedding = self.embedder
        )

        self._train_auto_encoder()
        self._train_ori_clas_sentence()
        
        # Classification loss for the generator, based on soft samples
        # Continuous softmax decoding, used in training
        # We will consider Gumbel-softmax decoding
        start_tokens = tf.ones_like(self.labels) * self.vocab.bos_token_id
        end_token = self.vocab.eos_token_id
        gumbel_helper = GumbelSoftmaxEmbeddingHelper(self.embedder.embedding, start_tokens, end_token, self.gamma)

        # for training
        self.soft_g_outputs_, _, self.soft_g_length_ = self.decoder(
            memory = self.enc_outputs_[:, 1:, :], 
            initial_state = self.enc_outputs_state_,
            helper = gumbel_helper,
        )

        # for eval
        self.g_outputs_, _, self.g_length_ = self.decoder(
            decoding_strategy='infer_greedy',
            memory = self.enc_outputs_[:, 1:, :],
            memory_sequence_length = self.sequence_length - 1,
            nitial_state = self.enc_outputs_state_,
            embedding = self.embedder, 
            start_tokens = start_tokens, 
            end_token = end_token
        )

        self._train_trans_clas_sentence()
    
    
    def _train_auto_encoder(self):
        """
        Use: 
            self.rephrase_encoder, self.rephrase_decoder
            self.g_outputs, self.text_ids

        Create:
            self.loss_g_ae
        """
        self.loss_g_ae = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels = self.text_ids[:, 1:],
            logits = self.g_outputs.logits,
            sequence_length = self.sequence_length - 1,
            average_across_timesteps = True,
            sum_over_timesteps = False
        )

    
    def _train_ori_clas_sentence(self):
        """Classification loss for the classifier
        Use: 
            self.classifier_sentence, self.clas_embedder
            self.text_ids

        Create:
            self.loss_d_clas_sentence, self.accu_d_sentence
        """
        clas_logits_sentence, clas_preds_sentence = self.classifier_sentence(
            inputs = self.clas_embedder(ids = self.text_ids[:, 1:]),
            sequence_length = self.sequence_length - 1
        )
        loss_d_clas_sentence = tf.nn.sigmoid_cross_entropy_with_logits(
            labels = tf.to_float(self.labels), 
            logits = clas_logits_sentence
        )
        self.loss_d_clas_sentence = tf.reduce_mean(loss_d_clas_sentence)
        self.accu_d_sentence = tx.evals.accuracy(
            labels = self.labels, 
            preds = clas_preds_sentence
        )

    
    def _train_trans_clas_sentence(self):
        """Classification loss for the tansferred generator
        Use: 
            self.rephrase_encoder, self.rephrase_decoder
            self.g_outputs_, self.clas_embedder, self.embedder

        Create:
            self.loss_g_clas_sentence, self.accu_g_sentence,
            self.accu_g_gdy_sentence
        """
        # Accuracy on soft samples, for training progress monitoring
        soft_logits_sentence, soft_preds_sentence = self.classifier_sentence(
            inputs = self.clas_embedder(soft_ids=self.soft_g_outputs_.sample_id),
            sequence_length = self.soft_g_length_
        )
        loss_g_clas_sentence = tf.nn.sigmoid_cross_entropy_with_logits(
            labels = tf.to_float(1 - self.labels), 
            logits = soft_logits_sentence
        )
        self.loss_g_clas_sentence = tf.reduce_mean(loss_g_clas_sentence)
        self.accu_g_sentence = tx.evals.accuracy(
            labels = 1 - self.labels, 
            preds = soft_preds_sentence
        )

        # Greedy decoding, used in eval, for training progress monitoring
        _, gdy_preds_sentence = self.classifier_sentence(
            inputs = self.clas_embedder(ids=self.g_outputs_.sample_id),
            sequence_length = self.g_length_
        )
        self.accu_g_gdy_sentence = tx.evals.accuracy(
            labels = 1 - self.labels,
            preds = gdy_preds_sentence
        )


    def _get_loss_train_op(self):
        # Aggregates losses
        self.loss_g = self.loss_g_ae + self.lambda_g_graph * self.loss_g_clas_graph + self.lambda_g_sentence * self.loss_g_clas_sentence
        self.loss_d = self.loss_d_clas_graph + self.loss_d_clas_sentence

        # Creates optimizers
        self.g_vars = collect_trainable_variables(
            [self.embedder, self.self_graph_encoder, self.label_connector, self.decoder])
        self.d_vars = collect_trainable_variables([self.clas_embedder, self.classifier_graph, self.classifier_sentence])

        self.train_op_g = get_train_op(
            self.loss_g, self.g_vars, hparams=self._hparams.opt)
        self.train_op_g_ae = get_train_op(
            self.loss_g_ae, self.g_vars, hparams=self._hparams.opt)
        self.train_op_d = get_train_op(
            self.loss_d, self.d_vars, hparams=self._hparams.opt)

        # Interface tensors
        self.losses = {
            "loss_g": self.loss_g,
            "loss_d": self.loss_d,
            "loss_g_ae": self.loss_g_ae,
            "loss_g_clas_graph": self.loss_g_clas_graph,
            "loss_g_clas_sentence": self.loss_g_clas_sentence,
            "loss_d_clas_graph": self.loss_d_clas_graph,
            "loss_d_clas_sentence": self.loss_d_clas_sentence,
        }
        self.metrics = {
            "accu_d_graph": self.accu_d_graph,
            "accu_d_sentence": self.accu_d_sentence,
            "accu_g_graph": self.accu_g_graph,
            "accu_g_sentence": self.accu_g_sentence,
            "accu_g_gdy_sentence": self.accu_g_gdy_sentence
        }
        self.train_ops = {
            "train_op_g": self.train_op_g,
            "train_op_g_ae": self.train_op_g_ae,
            "train_op_d": self.train_op_d
        }
        self.samples = {
            "original": self.text_ids[:, 1:],
            "transferred": self.g_outputs_.sample_id
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
        fetches_eval = {"batch_size": get_batch_size(self.text_ids)}
        fetches_eval.update(self.losses)
        fetches_eval.update(self.metrics)
        fetches_eval.update(self.samples)
        self.fetches_eval = fetches_eval



    
        
        
