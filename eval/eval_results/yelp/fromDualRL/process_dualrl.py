import pdb
import os
import shutil

datasets = ['BackTranslation_Pr', 'CrossAlignment_Shen', 'DeleteOnly_Li', 'DeleteRetrieve_Li', 'DualRL', 'Multidecoder_Fu', 'RetrieveOnly_Li', 'StyleEmbedding_Fu', 'TemplateBase_Li', 'UnpairedRL_Xu', 'UnsuperMT_Zhang']

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
for dataset_ in datasets:
    print('processing dataset: {}...'.format(dataset_))
    with open('{}/test.0-1.tsf'.format(dataset_), mode='r') as f:
        for line in f.readlines():
            line = line.split('\t')
            ori_ = replace_unk('{}\n'.format(line[0]), vocab)
            trans_ = line[1]
            out_dict[ori_] = trans_
    with open('{}/test.1-0.tsf'.format(dataset_), mode='r') as f:
        for line in f.readlines():
            line = line.split('\t')
            ori_ = replace_unk('{}\n'.format(line[0]), vocab)
            trans_ = line[1]
            out_dict[ori_] = trans_
    
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


            
 
