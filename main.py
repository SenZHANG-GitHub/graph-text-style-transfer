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

This is a simplified implementation of:

Toward Controlled Generation of Text, ICML2017
Zhiting Hu, Zichao Yang, Xiaodan Liang, Ruslan Salakhutdinov, Eric Xing

Download the data with the cmd:

$ python prepare_data.py

Train the model with the cmd:

$ python main.py --config config
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, too-many-locals, too-many-arguments, no-member

import os
import sys
import argparse
import importlib
import pdb
import logging
import numpy as np
import tensorflow as tf
import texar as tx

from models.GTAE_model import GTAE
from utils_data.multi_aligned_data_with_numpy import MultiAlignedNumpyData


# get config
flags = tf.flags
flags.DEFINE_string('config', 'config', 'The config to use.')
flags.DEFINE_string('out', 'tmp', 'The config to use.')
FLAGS = flags.FLAGS
config = importlib.import_module(FLAGS.config)
output_path = FLAGS.out
if output_path == 'none':
    raise ValueError('output path is not specified. E.g. python main.py --out output_path')

# get logger
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
logger_format_str = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
logger_format = logging.Formatter(logger_format_str)
logger_sh = logging.StreamHandler()
logger_sh.setFormatter(logger_format)
logger_th = logging.FileHandler('{}.log'.format(output_path), mode='w')
logger_th.setFormatter(logger_format)
logger.addHandler(logger_sh)
logger.addHandler(logger_th)

logger.info('config: {}.py'.format(FLAGS.config))

def _main(_):
    # Create output_path
    if os.path.exists(output_path):
        logger.error('output path {} already exists'.format(output_path))
        raise ValueError('output path {} already exists'.format(output_path))
    os.mkdir(output_path)
    os.mkdir('{}/src'.format(output_path))
    os.system('cp *.py {}/src'.format(output_path))
    os.system('cp models/*.py {}/src'.format(output_path))
    os.system('cp utils_data/*.py {}/src'.format(output_path))

    # clean sample_path and checkpoint_path before training 
    if tf.gfile.Exists(config.sample_path):
        tf.gfile.DeleteRecursively(config.sample_path)
    if tf.gfile.Exists(config.checkpoint_path):
        tf.gfile.DeleteRecursively(config.checkpoint_path)
    tf.gfile.MakeDirs(config.sample_path)
    tf.gfile.MakeDirs(config.checkpoint_path)
    
    # Data
    train_data = MultiAlignedNumpyData(config.train_data)
    val_data = MultiAlignedNumpyData(config.val_data)
    test_data = MultiAlignedNumpyData(config.test_data)
    vocab = train_data.vocab(0)

    # Each training batch is used twice: once for updating the generator and
    # once for updating the discriminator. Feedable data iterator is used for
    # such case.
    iterator = tx.data.FeedableDataIterator(
        {'train_g': train_data, 'train_d': train_data,
         'val': val_data, 'test': test_data})
    batch = iterator.get_next()

    # Model
    gamma = tf.placeholder(dtype=tf.float32, shape=[], name='gamma')
    lambda_g_graph = tf.placeholder(dtype=tf.float32, shape=[], name='lambda_g_graph')
    lambda_g_sentence = tf.placeholder(dtype=tf.float32, shape=[], name='lambda_g_sentence')
    
    if config.model_name == 'GTAE':
        model = GTAE(batch, vocab, gamma, lambda_g_graph, lambda_g_sentence, config.model)
    else:
        logger.error('config.model_name: {} is incorrect'.format(config.model_name))
        raise ValueError('config.model_name: {} is incorrect'.format(config.model_name))

    def _train_epoch(sess, gamma_, lambda_g_graph_, lambda_g_sentence_, epoch, verbose=True):
        avg_meters_d = tx.utils.AverageRecorder(size=10)
        avg_meters_g = tx.utils.AverageRecorder(size=10)

        step = 0
        while True:
            try:
                step += 1
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, 'train_d'),
                    gamma: gamma_,
                    lambda_g_graph: lambda_g_graph_,
                    lambda_g_sentence: lambda_g_sentence_
                }

                vals_d = sess.run(model.fetches_train_d, feed_dict=feed_dict)
                avg_meters_d.add(vals_d)

                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, 'train_g'),
                    gamma: gamma_,
                    lambda_g_graph: lambda_g_graph_,
                    lambda_g_sentence: lambda_g_sentence_
                }
                vals_g = sess.run(model.fetches_train_g, feed_dict=feed_dict)
                avg_meters_g.add(vals_g)

                if verbose and (step == 1 or step % config.display == 0):
                    logger.info('step: {}, {}'.format(step, avg_meters_d.to_str(4)))
                    logger.info('step: {}, {}'.format(step, avg_meters_g.to_str(4)))
                    sys.stdout.flush()

                if verbose and step % config.display_eval == 0:
                    iterator.restart_dataset(sess, 'val')
                    _eval_epoch(sess, gamma_, lambda_g_graph_, lambda_g_sentence_, epoch)

            except tf.errors.OutOfRangeError:
                logger.info('epoch: {}, {}'.format(epoch, avg_meters_d.to_str(4)))
                logger.info('epoch: {}, {}'.format(epoch, avg_meters_g.to_str(4)))
                sys.stdout.flush()
                break

    def _eval_epoch(sess, gamma_, lambda_g_graph_, lambda_g_sentence_, epoch, val_or_test='val'):
        avg_meters = tx.utils.AverageRecorder()

        while True:
            try:
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, val_or_test),
                    gamma: gamma_,
                    lambda_g_graph: lambda_g_graph_,
                    lambda_g_sentence: lambda_g_sentence_,
                    tx.context.global_mode(): tf.estimator.ModeKeys.EVAL
                }

                vals = sess.run(model.fetches_eval, feed_dict=feed_dict)

                batch_size = vals.pop('batch_size')

                # Computes BLEU
                samples = tx.utils.dict_pop(vals, list(model.samples.keys()))
                hyps = tx.utils.map_ids_to_strs(samples['transferred'], vocab)

                refs = tx.utils.map_ids_to_strs(samples['original'], vocab)
                refs = np.expand_dims(refs, axis=1)

                bleu = tx.evals.corpus_bleu_moses(refs, hyps)
                vals['bleu'] = bleu

                avg_meters.add(vals, weight=batch_size)

                # Writes samples
                tx.utils.write_paired_text(
                    refs.squeeze(), hyps,
                    os.path.join(config.sample_path, 'val.%d'%epoch),
                    append=True, mode='v')

            except tf.errors.OutOfRangeError:
                logger.info('{}: {}'.format(
                    val_or_test, avg_meters.to_str(precision=4)))
                break

        return avg_meters.avg()

    # Runs the logics
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        saver = tf.train.Saver(max_to_keep=None)
        if config.restore:
            logger.info('Restore from: {}'.format(config.restore))
            saver.restore(sess, config.restore)

        iterator.initialize_dataset(sess)

        gamma_ = 1.
        lambda_g_graph_ = 0.
        lambda_g_sentence_ = 0.
        for epoch in range(1, config.max_nepochs + 1):
            if epoch > config.pretrain_nepochs:
                # Anneals the gumbel-softmax temperature
                gamma_ = max(0.001, gamma_ * config.gamma_decay)
                lambda_g_graph_ = config.lambda_g_graph
                lambda_g_sentence_ = config.lambda_g_sentence
            logger.info('gamma: {}, lambda_g_graph: {}, lambda_g_sentence: {}'.format(gamma_, lambda_g_graph_, lambda_g_sentence_))

            # Train
            iterator.restart_dataset(sess, ['train_g', 'train_d'])
            _train_epoch(sess, gamma_, lambda_g_graph_, lambda_g_sentence_, epoch)

            # Val
            iterator.restart_dataset(sess, 'val')
            _eval_epoch(sess, gamma_, lambda_g_graph_, lambda_g_sentence_, epoch, 'val')

            saver.save(
                sess, os.path.join(config.checkpoint_path, 'ckpt'), epoch)

            # Test
            iterator.restart_dataset(sess, 'test')
            _eval_epoch(sess, gamma_, lambda_g_graph_, lambda_g_sentence_, epoch, 'test')
    
    logger.info('tensorflow training process finished successlly!')
    if not os.path.exists('{}.log'.format(output_path)):
        logger.error('cannot find {}.log'.format(output_path))
    else:
        os.system('mv {}.log {}/'.format(output_path, output_path))
    if not os.path.exists('samples'):
        logger.error('cannot find samples/')
    else:
        os.system('mv samples/ {}/'.format(output_path))

if __name__ == '__main__':
    tf.app.run(main=_main)
        
        
