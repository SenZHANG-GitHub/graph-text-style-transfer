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
"""Text style transfer Under Linguistic Constraints
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
import pdb


class GTAE(object):
    """Control  
    """
    def __init__(self, inputs, vocab, gamma, lambda_t_graph, lambda_t_sentence, ablation, hparams=None):
        self._hparams = tx.HParams(hparams, None)
        self._prepare_inputs(inputs, vocab, gamma, lambda_t_graph, lambda_t_sentence, ablation),
        self._build_model()
    

    def _prepare_inputs(self, inputs, vocab, gamma, lambda_t_graph, lambda_t_sentence, ablation):
        self.vocab = vocab
        self.gamma = gamma
        self.lambda_t_graph = lambda_t_graph
        self.lambda_t_sentence = lambda_t_sentence
        
        # possible ablation: SGT-I, CGT-I, SGT-CGT-I, c-clas-g-only, c-clas-s-only
        if ablation not in ['full', 'SGT-I', 'CGT-I', 'SGT-CGT-I', 'c-clas-g-only', 'c-clas-s-only']:
            raise ValueError('--ablation {} is not supported'.format(ablation))
        self.ablation = ablation

        # the first token is the BOS token
        self.text_ids = inputs['text_ids']
        self.sequence_length = inputs['length']
        self.labels = inputs['labels']
        
        self.inputs = inputs

        enc_shape = tf.shape(self.text_ids)
        adjs = tf.to_int32(tf.reshape(inputs['adjs'], [-1,17,17]))
        self.adjs = adjs[:, :enc_shape[1], :enc_shape[1]]
        
        # 'identities' is only given in config_yelp
        if ablation in ['SGT-I', 'CGT-I', 'SGT-CGT-I']:
            if 'identities' in inputs.keys():
                identities = tf.to_int32(tf.reshape(inputs['identities'], [-1,17,17]))
                self.identities = identities[:, :enc_shape[1], :enc_shape[1]]
            else:
                raise ValueError('identities.tfrecords must be given in config if using --ablation SGT-I/CGT-I/SGT-CGT-I')


    def _build_model(self):
        """Builds the model.
        """
        self._prepare_modules()
        self._build_self_graph_encoder()
        self._build_cross_graph_encoder()
        self._get_loss_train_op()
    
    
    def _prepare_modules(self):
        """Prepare necessary modules
        """
        self.embedder = WordEmbedder(
            vocab_size = self.vocab.size, 
            hparams=self._hparams.embedder
        )
        self.clas_embedder = WordEmbedder(
            vocab_size = self.vocab.size, 
            hparams = self._hparams.embedder
        )
        self.label_connector = MLPTransformConnector(self._hparams.dim_c)

        self.self_graph_encoder = SelfGraphTransformerEncoder(hparams=self._hparams.encoder)
        self.cross_graph_encoder = CrossGraphTransformerFixedLengthDecoder(
            vocab_size = self.vocab.size,
            tau = self.gamma,
            hparams = self._hparams.encoder
        )

        self.classifier_graph = Conv1DClassifier(hparams=self._hparams.classifier)
        self.classifier_sentence = Conv1DClassifier(hparams=self._hparams.classifier)

        self.rephrase_encoder = UnidirectionalRNNEncoder(hparams=self._hparams.rephrase_encoder)
        self.rephrase_decoder =DynamicAttentionRNNDecoder(
            memory_sequence_length = self.sequence_length - 1,
            cell_input_fn = lambda inputs, attention: inputs,
            vocab_size = self.vocab.size,
            hparams = self._hparams.rephrase_decoder
        )


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
        if self.ablation in ['SGT-I', 'SGT-CGT-I']: 
            self.enc_outputs = self.self_graph_encoder(
                inputs = self.embedding_text_ids, 
                sequence_length = self.sequence_length, 
                adjs = self.identities
            )
            # encoding output_ for transferred style
            self.enc_outputs_ = self.self_graph_encoder(
                inputs = self.embedding_text_ids_, 
                sequence_length = self.sequence_length, 
                adjs = self.identities
            )
        else:
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
            inputs = self.enc_outputs[:, 1:, :],
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

    
    def _build_cross_graph_encoder(self):
        """
        Use: 
            self.cross_graph_encoder, self.enc_outputs
            self.pre_embedding_text_ids

        Create:
            self.g_outputs, self.g_outputs_
        """
        if self.ablation in ['CGT-I', 'SGT-CGT-I']:
            self.g_outputs = self.cross_graph_encoder(
                inputs = self.enc_outputs[:, 1:, :], 
                memory = self.pre_embedding_text_ids,
                sequence_length = self.sequence_length - 1, 
                memory_sequence_length = self.sequence_length-1,
                adjs = self.identities[:, 1:, 1:],
                encoder_output = True
            )
            self.g_outputs_ = self.cross_graph_encoder(
                inputs = self.enc_outputs_[:, 1:, :],
                memory = self.pre_embedding_text_ids,
                sequence_length = self.sequence_length - 1,
                memory_sequence_length = self.sequence_length - 1,
                adjs = self.identities[:, 1:, 1:],
                encoder_output = True
            )
        else:
            # Auto-encoding loss for G
            # The first token that represents BOS/CLS is removed
            # Currently use the same sequence_length and memory_sequence_length
            # Later we may consider use the CLS flag in embedding_text_ids to guide the generation
            self.g_outputs = self.cross_graph_encoder(
                inputs = self.enc_outputs[:, 1:, :], 
                memory = self.pre_embedding_text_ids,
                sequence_length = self.sequence_length - 1, 
                memory_sequence_length = self.sequence_length-1,
                adjs = self.adjs[:, 1:, 1:],
                encoder_output = True
            )
            
            # Classification loss for the generator, based on soft samples
            # Continuous softmax decoding, used in training
            # We will consider Gumbel-softmax decoding
            self.g_outputs_ = self.cross_graph_encoder(
                inputs = self.enc_outputs_[:, 1:, :],
                memory = self.pre_embedding_text_ids,
                sequence_length = self.sequence_length - 1,
                memory_sequence_length = self.sequence_length - 1,
                adjs = self.adjs[:, 1:, 1:],
                encoder_output = True
            )

        self._train_auto_encoder()
        self._train_ori_clas_sentence()
        self._train_trans_clas_sentence()
    
    
    def _train_auto_encoder(self):
        """
        Use: 
            self.rephrase_encoder, self.rephrase_decoder
            self.g_outputs, self.text_ids

        Create:
            self.loss_g_ae
        """
        rephrase_enc, rephrase_state = self.rephrase_encoder(
            self.g_outputs, 
            sequence_length = self.sequence_length - 1
        )
        rephrase_outputs, _, _ = self.rephrase_decoder(
            initial_state = rephrase_state,
            memory = rephrase_enc, # embedder(inputs['text_ids'][:, 1:]),
            sequence_length = self.sequence_length - 1,
            inputs = self.text_ids,
            embedding = self.embedder
        )
        self.loss_g_ae = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels = self.text_ids[:, 1:],
            logits = rephrase_outputs.logits,
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
        # Gumbel-softmax decoding, used in training
        start_tokens = tf.ones_like(self.labels) * self.vocab.bos_token_id
        end_token = self.vocab.eos_token_id
        gumbel_helper = GumbelSoftmaxEmbeddingHelper(
            self.embedder.embedding, 
            start_tokens, 
            end_token, 
            self.gamma
        )
        rephrase_enc_, rephrase_state_ = self.rephrase_encoder(
            self.g_outputs_, 
            sequence_length = self.sequence_length - 1
        )

        # Accuracy on soft samples, for training progress monitoring
        soft_rephrase_outputs_, _, soft_rephrase_length_ = self.rephrase_decoder(
            memory = rephrase_enc_, 
            helper = gumbel_helper,
            initial_state = rephrase_state_
        )
        soft_logits_sentence, soft_preds_sentence = self.classifier_sentence(
            inputs = self.clas_embedder(soft_ids=soft_rephrase_outputs_.sample_id),
            sequence_length = soft_rephrase_length_
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
        self.rephrase_outputs_, _, rephrase_length_ = self.rephrase_decoder(
            decoding_strategy = 'infer_greedy',
            memory = rephrase_enc_,
            initial_state = rephrase_state_,
            embedding = self.embedder,
            start_tokens = start_tokens,
            end_token = end_token
        ) 
        _, gdy_preds_sentence = self.classifier_sentence(
            inputs = self.clas_embedder(ids=self.rephrase_outputs_.sample_id),
            sequence_length = rephrase_length_
        )
        self.accu_g_gdy_sentence = tx.evals.accuracy(
            labels = 1 - self.labels,
            preds = gdy_preds_sentence
        )


    def _get_loss_train_op(self):
        # Aggregates losses
        self.loss_g = self.loss_g_ae + self.lambda_t_graph * self.loss_g_clas_graph + self.lambda_t_sentence * self.loss_g_clas_sentence
        # possible ablation: SGT-I, CGT-I, SGT-CGT-I, c-clas-g-only, c-clas-s-only
        if self.ablation == 'c-clas-g-only':
            self.loss_d = self.loss_d_clas_graph
        elif self.ablation == 'c-clas-s-only':
            self.loss_d = self.loss_d_clas_sentence
        else:
            self.loss_d = self.loss_d_clas_graph + self.loss_d_clas_sentence

        # Creates optimizers
        self.g_vars = collect_trainable_variables(
            [self.embedder, self.self_graph_encoder, self.label_connector, self.cross_graph_encoder, self.rephrase_encoder, self.rephrase_decoder])
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
            "transferred": self.rephrase_outputs_.sample_id
        }

        self.fetches_train_g = {
            "loss_g": self.train_ops["train_op_g"],
            "loss_g_ae": self.losses["loss_g_ae"],
            "loss_g_clas_graph": self.losses["loss_g_clas_graph"],
            "loss_g_clas_sentence": self.losses["loss_g_clas_sentence"],
            "accu_g_graph": self.metrics["accu_g_graph"],
            "accu_g_sentence": self.metrics["accu_g_sentence"],
            "accu_g_gdy_sentence": self.metrics["accu_g_gdy_sentence"]
            # 'adjs': self.adjs,
            # 'identities': self.identities
        }
        self.fetches_train_d = {
            "loss_d": self.train_ops["train_op_d"],
            "loss_d_clas_graph": self.losses["loss_d_clas_graph"],
            "loss_d_clas_sentence": self.losses["loss_d_clas_sentence"],
            "accu_d_graph": self.metrics["accu_d_graph"],
            "accu_d_sentence": self.metrics["accu_d_sentence"]
            # 'adjs': self.adjs,
            # 'identities': self.identities
        }
        fetches_eval = {"batch_size": get_batch_size(self.text_ids)}
        fetches_eval.update(self.losses)
        fetches_eval.update(self.metrics)
        fetches_eval.update(self.samples)
        self.fetches_eval = fetches_eval



    
        
        
