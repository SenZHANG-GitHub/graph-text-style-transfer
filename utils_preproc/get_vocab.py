import pdb
import texar as tx

dataset = 'amazon'
data_path = '/home/dm/Documents/text_generation/GraphTextTransfer/data/{}'.format(dataset)

train_text = '{}/{}.short15.train.text'.format(data_path, dataset)
count_threshold = 5

vocab_ori = tx.data.make_vocab(train_text, return_count=True)
vocab_filter = [x for _i, x in enumerate(vocab_ori[0]) if vocab_ori[1][_i] >= count_threshold]

with open('{}/vocab_{}'.format(data_path, dataset), 'w') as f:
    f.write('\n'.join(vocab_filter))

