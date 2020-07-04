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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import importlib
import tensorflow as tf
import texar as tx
import numpy as np

# parameters for textcnn
# Word embedding
emb = {
    "dim": 300
}

# Classifier
textcnn_clas = {
    "num_conv_layers": 1,
    "filters": 100,
    "kernel_size": [3, 4, 5],
    "conv_activation": "relu",
    "pooling": "MaxPooling1D",
    "num_dense_layers": 0,
    "dropout_conv": [1],
    "dropout_rate": 0.5,
    "num_classes": 2
}

# Optimization
# Just use the default config, e.g., Adam Optimizer
opt = {}


def test_clas(test_data, result_path, npz_path, ckpt_path, as_text):
    # Data
    test_data = tx.data.MultiAlignedData(test_data)
    iterator = tx.data.DataIterator({'test': test_data})
    batch = iterator.get_next()

    # Model architecture
    vocab = test_data.vocab('x')
    embedder = tx.modules.WordEmbedder(
        vocab_size=vocab.size, hparams=emb)
    classifier = tx.modules.Conv1DClassifier(textcnn_clas)
    logits, pred = classifier(embedder(batch['x_text_ids']))

    probs = tf.nn.softmax(logits)

    # Losses & train ops
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=batch['y'], logits=logits)
    accu = tx.evals.accuracy(batch['y'], pred)

    batch_x_ids = batch['x_text_ids']
    batch_y = batch['y']

    train_op = tx.core.get_train_op(loss, hparams=opt)

    def _run_epoch(sess, mode, epoch=0, verbose=False):
        is_train = tx.utils.is_train_mode_py(mode)

        fetches = {
            "accu": accu,
            "logits": logits,
            "probs": probs,
            "pred": pred,
            "batch_size": tx.utils.get_batch_size(batch['y']),
            "batch_x_ids": batch_x_ids,
            "batch_y": batch_y
        }
        if is_train:
            fetches["train_op"] = train_op
        feed_dict = {tx.context.global_mode(): mode}

        cum_accu = 0.
        nsamples = 0
        step = 0
        distribution = []
        label_list = []
        ftext = open(result_path, 'w')
        while True:
            try:
                rets = sess.run(fetches, feed_dict)
                step += 1

                accu_ = rets['accu']
                cum_accu += accu_ * rets['batch_size']
                nsamples += rets['batch_size']

                out_x = tx.utils.map_ids_to_strs(rets['batch_x_ids'], vocab)
                out_x = as_text('\n'.join(out_x)).split('\n')
                out_y = rets['batch_y']

                out_pred = rets['pred']
                out_logits = rets['logits']
                out_probs = rets['probs']

                distribution.extend(out_probs)
                label_list.extend(rets['batch_y'])

                for _ibatch in range(rets['batch_size']):
                    ftext.write('label: {} | pred: {} | p0: {} | p1: {} | {}\n'.format(out_y[_ibatch], out_pred[_ibatch], out_probs[_ibatch][0], out_probs[_ibatch][1], out_x[_ibatch]))

                if verbose and (step == 1 or step % 100 == 0):
                    tf.logging.info(
                        "epoch: {0:2} step: {1:4} accu: {2:.4f}"
                        .format(epoch, step, accu_))
            except tf.errors.OutOfRangeError:
                break
        np.savez(npz_path, textcnn=distribution, label=label_list)
        ftext.close()
        return cum_accu / nsamples

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        saver = tf.train.Saver(max_to_keep=None)
        if os.path.isdir('/'.join(ckpt_path.split('/')[:-1])):
            print('restore {}'.format(ckpt_path))
            saver.restore(sess, ckpt_path)

            iterator.switch_to_dataset(sess, 'test')
            test_accu = _run_epoch(sess, tf.estimator.ModeKeys.EVAL)
            tf.logging.info('test accu: {0:.4f}'.format(test_accu))
            return test_accu
        else:
            raise ValueError('Please give correct ckpt_path')


def generate_style_distr(dataset, model, style, restore_ckpt='ckpt-1'):
    """ e.g.
    Input: (1) dataset: yelp (2) model: GTAE-alfa-XX (3) style: trans or ori
    Output: 
    ->    '../eval_results/yelp/GTAE-alfa-XX/trans_distribution.npz' 
    -> or '../eval_results/yelp/GTAE-alfa-XX/ori_distribution.npz' 
    """
    as_text = tf.compat.as_text
    # pylint: disable=invalid-name, too-many-locals

    # e.g. result_path: 'eval_results/yelp/GTAE-alfa-XX/trans_distribution.text' just for display
    # e.g. npz_path:    'eval_results/yelp/GTAE-alfa-XX/trans_distribution'
    result_path = 'eval_results/{}/{}/{}_distribution.text'.format(dataset, model, style)
    npz_path = 'eval_results/{}/{}/{}_distribution'.format(dataset, model, style)

    # e.g. ckpt_path: 'eval_models/clas_yelp/ckpt-1
    ckpt_path = 'classifier/ckpt_{}/{}'.format(dataset, restore_ckpt)
    vocab_path = 'classifier/vocab_{}'.format(dataset)

    test_data = {
        "batch_size": 50,
        "shuffle": False,
        "datasets": [
            {   
                # e.g. files: '../eval_results/yelp/GTAE-alfa-XX/trans.text'
                "files": "eval_results/{}/{}/{}.text".format(dataset, model, style),
                "vocab_file": vocab_path,
                # Discards samples with length > 56
                "max_seq_length": 56,
                "length_filter_mode": "discard",
                # Do not append BOS/EOS tokens to the sentences
                "bos_token": "",
                "eos_token": "",
                "data_name": "x"
            },
            {
                # e.g. files: '../eval_results/yelp/GTAE-alfa-XX/trans.label'
                "files": "eval_results/{}/{}/{}.label".format(dataset, model, style),
                "data_type": "int",
                "data_name": "y"
            }
        ]
    }
    tf.logging.set_verbosity(tf.logging.INFO)
    test_accu = test_clas(test_data, result_path, npz_path, ckpt_path, as_text)
    tf.reset_default_graph()
    return test_accu
