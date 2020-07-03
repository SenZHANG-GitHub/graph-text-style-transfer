# import tensorflow as tf
import json
import numpy as np 
import tensorflow as tf
import time
import argparse
from tqdm import tqdm
import pdb

args = argparse.ArgumentParser(description='Process dir name')
args.add_argument('json_name', type = str, help = 'The json file that extracts the relations!')
args.add_argument('tfrecord_adjs_name', type = str, help = 'The tfrecord file that saves relations infos!')
args.add_argument('tfrecord_identities_name', type = str, help = 'The tfrecord file that saves identity matrix with the same shape as adjs')
args = args.parse_args()

class ReadDataset(object):
    def __init__(self):
        pass

    def read_save_dataset(self, json_name, tf_adjs_name, tf_identities_name):
        # Data format:
        # {'sent': 'i was sadly mistaken .', 
        #  'relations': [[0, 4, 'root'], [4, 1, 'nsubjpass'], [4, 2, 'auxpass'], [4, 3, 'advmod']], 
        #  'words': {'1': 'i', '4': 'mistaken', '2': 'was', '3': 'sadly', '0': 'root'}}

        sent_length = []
        print('calculating max_length from json data: {}'.format(json_name))
        with open(json_name ,"r") as f:
            for line in tqdm(f.readlines()):
                each = json.loads(line)           
                sent_words = each['sent'].split()
                sent_length.append(len(sent_words)+2)
        max_sent_length = max(sent_length)
        print('max length: {}'.format(max(sent_length)))   # including <BOS>, <EOS>
        print('min length: {}'.format(min(sent_length)))   # including <BOS>, <EOS>

        # relation matrix
        print('read json and save tfrecords data: {} and {}'.format(tf_adjs_name, tf_identities_name))
        with tf.python_io.TFRecordWriter(tf_adjs_name) as writer_adjs:
            with tf.python_io.TFRecordWriter(tf_identities_name) as writer_identities:
                with open(json_name ,"r") as f:
                    for line in tqdm(f.readlines()):
                        each = json.loads(line)           
                        true_length = len(each['sent'].split()) + 1
                        length = max_sent_length
                        each_relations = [([0] * length) for i in range(length)]
                        for i in range(length):
                            if i==0:
                                for j in range(true_length):
                                    each_relations[0][j] = 1
                            elif i<true_length:
                                each_relations[i][0] = 1

                        for e in each['relations']:
                            if e[0]!=0 and e[1]!=0:
                                each_relations[e[0]][e[1]] = 1  
                                each_relations[e[1]][e[0]] = 1  
                                
                        tmp_relation = np.array(each_relations)
                        # fix the bug of diagonal items which should be 1
                        for i_diag in range(true_length):
                            tmp_relation[i_diag, i_diag] = 1

                        features = tf.train.Features(
                            feature = {
                                'adjs': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tmp_relation.astype(np.int32).tostring()]))  
                            }
                        )
                        example = tf.train.Example(features = features)
                        serialized = example.SerializeToString()
                        writer_adjs.write(serialized)
                        
                        identity_matrix = np.zeros_like(tmp_relation)
                        for i_diag in range(true_length):
                            identity_matrix[i_diag, i_diag] = 1
                            identity_matrix[0, i_diag] = 1
                            identity_matrix[i_diag, 0] = 1
                                                    
                        identity_features = tf.train.Features(
                            feature = {
                                'identities': tf.train.Feature(bytes_list = tf.train.BytesList(value = [identity_matrix.astype(np.int32).tostring()]))  
                            }
                        )
                        identity_example = tf.train.Example(features = identity_features)
                        identity_serialized = identity_example.SerializeToString()
                        writer_identities.write(identity_serialized)


def _parse_function(example_proto):
    features = {
            'adjs': tf.FixedLenFeature((), tf.string)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    data = tf.decode_raw(parsed_features['adjs'], tf.int32)
    return data

def load_tfrecords(srcfile):
    sess = tf.Session()

    dataset = tf.data.TFRecordDataset(srcfile)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(64)

    iterator = dataset.make_one_shot_iterator()
    next_data = iterator.get_next()

    while True:
        try:
            data = sess.run(next_data)
            for each in data:
                print(each)
            print(np.size(data))
            time.sleep(3)
        except tf.errors.OutOfRangeError:
            break


if __name__ == '__main__':
    json_name = args.json_name
    tf_adjs_name = args.tfrecord_adjs_name
    tf_identities_name = args.tfrecord_identities_name

    read_dataset = ReadDataset()
    read_dataset.read_save_dataset(json_name, tf_adjs_name, tf_identities_name)

