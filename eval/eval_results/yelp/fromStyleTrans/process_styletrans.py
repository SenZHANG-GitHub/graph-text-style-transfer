import pdb
import os
import shutil

def read_dataset(filepath):
    tmplist = []
    with open(filepath, mode='r') as f:
        for line in f.readlines():
            tmplist.append(line)
    return tmplist

def replace_unk(text, vocab):
    new_text = []
    for word in text.split(' '):
        word = word.replace('\n', '')
        if word not in vocab:
            new_text.append('<UNK>')
        else:
            new_text.append(word)
    return '{}\n'.format(' '.join(new_text))

vocab = read_dataset('../../../../data/yelp/vocab_yelp')
for iv, vocab_ in enumerate(vocab):
    vocab[iv] = vocab_[:-1]

# Read global ori_common.text
ori_common_dict = dict()
ori_common_texts = []
ori_common_labels = []
with open('../ori_common.text', mode='r') as f:
    for line in f.readlines():
        ori_common_texts.append(line)

with open('../ori_common.label', mode='r') as f:
    for line in f.readlines():
        ori_common_labels.append(line)

for text_, label_ in zip(ori_common_texts, ori_common_labels):
    ori_common_dict[text_] = int(label_.strip())
    
for suffix in ['cond', 'multi']:
    dataset_ = 'StyleTransformer-{}'.format(suffix)
    print('processing dataset: {}...'.format(dataset_))

    ori_0 = []
    ori_1 = []
    with open('test.neg', mode='r') as f:
        for line in f.readlines():
            ori_0.append(replace_unk(line, vocab))
    with open('test.pos', mode='r') as f:
        for line in f.readlines():
            ori_1.append(replace_unk(line, vocab))

    out_dict = dict()

    with open('output.0.ours_{}'.format(suffix), mode='r') as f:
        for i_, line in enumerate(f.readlines()):
            out_dict[ori_0[i_]] = line
            
    with open('output.1.ours_{}'.format(suffix), mode='r') as f:
        for i_, line in enumerate(f.readlines()):
            out_dict[ori_1[i_]] = line
            
    if os.path.isdir('../{}'.format(dataset_)):
        shutil.rmtree('../{}'.format(dataset_))
    os.mkdir('../{}'.format(dataset_))

    fori_t = open('../{}/ori.text'.format(dataset_), mode='w')
    fori_l = open('../{}/ori.label'.format(dataset_), mode='w')
    ftrans_t = open('../{}/trans.text'.format(dataset_), mode='w')
    ftrans_l = open('../{}/trans.label'.format(dataset_), mode='w')
    for text_ in ori_common_texts:
        if text_ not in out_dict.keys():
            pdb.set_trace()
            raise ValueError('{} is not in the common texts'.format(text_))
        fori_t.write(text_)
        fori_l.write('{}\n'.format(ori_common_dict[text_]))
        ftrans_t.write(out_dict[text_])
        ftrans_l.write('{}\n'.format(1 - ori_common_dict[text_]))

    fori_t.close()
    fori_l.close()
    ftrans_t.close()
    ftrans_l.close()


            
 
