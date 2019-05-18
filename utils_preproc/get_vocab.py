import pdb
import texar as tx
import argparse

args = argparse.ArgumentParser(description='Process dir name')
args.add_argument('train_file', type = str, help = 'The training text file on which the vocab is built')
args.add_argument('vocab_file', type = str, help = 'Filename of output vocab file')
args = args.parse_args()

train_text = args.train_file
count_threshold = 5

vocab_ori = tx.data.make_vocab(train_text, return_count=True)
vocab_filter = [x for _i, x in enumerate(vocab_ori[0]) if vocab_ori[1][_i] >= count_threshold]

with open(args.vocab_file, 'w') as f:
    f.write('\n'.join(vocab_filter))

