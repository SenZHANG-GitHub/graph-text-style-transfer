import pdb
import os
import shutil

datasets = ['Seq2SentiSeq']

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

out_dict = dict()
out_dict['1'] = dict() # style transfer intensity: 0.1
out_dict['3'] = dict() # style transfer intensity: 0.3
out_dict['5'] = dict() # style transfer intensity: 0.5
out_dict['7'] = dict() # style transfer intensity: 0.7
out_dict['9'] = dict() # style transfer intensity: 0.9

for dataset_ in datasets:
    print('processing dataset: {}...'.format(dataset_))
    with open('{}/test_has_source.tsf'.format(dataset_), mode='r') as f:
        cnt = 0
        for line in f.readlines():
            line = line.split('\t')
            ori_ = replace_unk('{}\n'.format(line[0]), vocab)
            trans_ = '{}\n'.format(line[1])
            if cnt % 5 == 0:
                out_dict['1'][ori_] = trans_
            elif cnt % 5 == 1:
                out_dict['3'][ori_] = trans_
            elif cnt % 5 == 2:
                out_dict['5'][ori_] = trans_
            elif cnt % 5 == 3:
                out_dict['7'][ori_] = trans_
            elif cnt % 5 == 4:
                out_dict['9'][ori_] = trans_
            cnt += 1
            
    for key_ in ['1', '3', '5', '7', '9']:
        if os.path.isdir('../{}-{}'.format(dataset_, key_)):
            shutil.rmtree('../{}-{}'.format(dataset_, key_))
        os.mkdir('../{}-{}'.format(dataset_, key_))
        
        fori_t = open('../{}-{}/ori.text'.format(dataset_, key_), mode='w')
        fori_l = open('../{}-{}/ori.label'.format(dataset_, key_), mode='w')
        ftrans_t = open('../{}-{}/trans.text'.format(dataset_, key_), mode='w')
        ftrans_l = open('../{}-{}/trans.label'.format(dataset_, key_), mode='w')
        cmm_cnt = 0
        for text_ in ori_common_texts:
            if text_ not in out_dict[key_].keys():
                continue
                pdb.set_trace()
                raise ValueError('{} is not in the common texts'.format(text_))
            cmm_cnt += 1
            fori_t.write(text_)
            fori_l.write('{}\n'.format(ori_common_dict[text_]))
            ftrans_t.write(out_dict[key_][text_])
            ftrans_l.write('{}\n'.format(1 - ori_common_dict[text_]))
        print('number of texts for {}-{}: {}'.format(dataset_, key_, cmm_cnt))
        
        fori_t.close()
        fori_l.close()
        ftrans_t.close()
        ftrans_l.close()


            
 
