import pdb
import os
import shutil

datasets = ['IMaT']

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
out_dict['ca_IMaT'] = dict() # style transfer intensity: 0.1
out_dict['multi_decoder_IMaT'] = dict() # style transfer intensity: 0.3
out_dict['style_emb_IMaT'] = dict() # style transfer intensity: 0.5
out_dict['dar_IMaT'] = dict() # style transfer intensity: 0.7
out_dict['IMaT'] = dict() # style transfer intensity: 0.9

for dataset_ in datasets:
    print('processing dataset: {}...'.format(dataset_))
    for tmpind_ in [0, 1]:
        with open('all_model_outputs.{}'.format(tmpind_), mode='r') as f:
            cnt = 0
            ori_ = ''
            for line in f.readlines():
                line = line.split('\t')
                flag = line[0][:-1]
                sentence = replace_unk(line[1], vocab)
                if cnt % 6 == 0:
                    if flag != 'src': raise ValueError('error')
                    ori_ = sentence
                elif cnt % 6 == 1:
                    if flag != 'ca': raise ValueError('error')
                    out_dict['ca_IMaT'][ori_] = sentence
                elif cnt % 6 == 2:
                    if flag != 'multi_decoder': raise ValueError('error')
                    out_dict['multi_decoder_IMaT'][ori_] = sentence
                elif cnt % 6 == 3:
                    if flag != 'style_emb': raise ValueError('error')
                    out_dict['style_emb_IMaT'][ori_] = sentence
                elif cnt % 6 == 4:
                    if flag != 'dar': raise ValueError('error')
                    out_dict['dar_IMaT'][ori_] = sentence
                elif cnt % 6 == 5:
                    if flag != 'ours': raise ValueError('error')
                    out_dict['IMaT'][ori_] = sentence
                cnt += 1
            
    for key_ in ['ca_IMaT', 'multi_decoder_IMaT', 'style_emb_IMaT', 'dar_IMaT', 'IMaT']:
        if os.path.isdir('../{}'.format(key_)):
            shutil.rmtree('../{}'.format(key_))
        os.mkdir('../{}'.format(key_))
        
        fori_t = open('../{}/ori.text'.format(key_), mode='w')
        fori_l = open('../{}/ori.label'.format(key_), mode='w')
        ftrans_t = open('../{}/trans.text'.format(key_), mode='w')
        ftrans_l = open('../{}/trans.label'.format(key_), mode='w')
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
        print('number of texts for {}: {}'.format(key_, cmm_cnt))
        
        fori_t.close()
        fori_l.close()
        ftrans_t.close()
        ftrans_l.close()


            
 
