"""Config
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

import copy
import texar as tx

max_nepochs = 15 # Total number of training epochs
                 # (including pre-train and full-train)
pretrain_nepochs = 10 # Number of pre-train epochs (training as autoencoder)
display = 500  # Display the training results every N training steps.
display_eval = 1e10 # Display the dev results every N training steps (set to a
                    # very large value to disable it).
sample_path = './samples'
checkpoint_path = './checkpoints'
restore = ''   # Model snapshot to restore from

model_name = 'GTAE'

lambda_g_graph = 0.02    # Weight of the graph classification loss
lambda_g_sentence = 0.02 # Weight of the sentence classification loss
gamma_decay = 0.5 # Gumbel-softmax temperature anneal rate

max_sequence_length = 15

train_data = {
    'batch_size': 64,
    #'seed': 123,
    'datasets': [
        {
            'files': './data/yelp/sentiment.train.text',
            'vocab_file': './data/yelp/vocab_yelp',
            'data_name': ''
        },
        {
            'files': './data/yelp/sentiment.train.labels',
            'data_type': 'int',
            'data_name': 'labels'
        },
        {
            'files': './data/yelp/sentiment.train_adjs.tfrecords',
            'data_type': 'tf_record',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length + 2, max_sequence_length + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        }
    ],
    'name': 'train'
}

val_data = copy.deepcopy(train_data)
val_data['datasets'][0]['files'] = './data/yelp/sentiment.dev.text'
val_data['datasets'][1]['files'] = './data/yelp/sentiment.dev.labels'
val_data['datasets'][2]['files'] = './data/yelp/sentiment.dev_adjs.tfrecords'

test_data = copy.deepcopy(train_data)
test_data['datasets'][0]['files'] = './data/yelp/sentiment.test.text'
test_data['datasets'][1]['files'] = './data/yelp/sentiment.test.labels'
test_data['datasets'][2]['files'] = './data/yelp/sentiment.test_adjs.tfrecords'

dim_hidden = 512
model = {
    'dim_c': dim_hidden,
    'embedder': {
        'dim': dim_hidden,
    },
    'encoder': {
        'num_blocks': 2,
        'dim': dim_hidden,
        'use_bert_config': False,
        'embedding_dropout': 0.1,
        'residual_dropout': 0.1,
        'graph_multihead_attention': {
            'name': 'multihead_attention',
            'num_units': dim_hidden,
            'output_dim': dim_hidden,
            'num_heads': 8,
            'dropout_rate': 0.1,
            'output_dim': dim_hidden,
            'use_bias': False,
        },
        'initializer': None,
        'name': 'graph_transformer_encoder',
    },
    'rephrase_encoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': dim_hidden
            },
            'dropout': {
                'input_keep_prob': 0.5
            }
        }
    },
    'rephrase_decoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': dim_hidden,
            },
            'dropout': {
                'input_keep_prob': 0.5,
                'output_keep_prob': 0.5
            },
        },
        'attention': {
            'type': 'DynamicBahdanauAttention',
            'kwargs': {
                'num_units': dim_hidden,
            },
            'attention_layer_size': dim_hidden,
        },
        'max_decoding_length_train': 21,
        'max_decoding_length_infer': 20,
    },
    'classifier': {
        'kernel_size': [3, 4, 5],
        'filters': 128,
        'other_conv_kwargs': {'padding': 'same'},
        'dropout_conv': [1],
        'dropout_rate': 0.5,
        'num_dense_layers': 0,
        'num_classes': 1
    },
    'opt': {
        'optimizer': {
            'type':  'AdamOptimizer',
            'kwargs': {
                'learning_rate': 5e-4,
            },
        },
    },
}
