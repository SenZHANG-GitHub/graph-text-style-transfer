"""EVALUATION OF BLEU SCORE
   
"""
working_dir_ = '/home/dm/Documents/text_generation/style-transfer-model-evaluation-master/classifiers'

import texar as tx 
import numpy as np
import pdb
import os
os.chdir(working_dir_)

def get_human_ref_dict(dataset_):
    ref_dict = dict()
    if dataset_ == "yelp_small":
        for _i in [0,1]:
            with open('{}/data/yelp_small/reference.{}'.format(working_dir_, _i), 'r') as ftmp:
                for line in ftmp.readlines():
                    tmp_pair = line.strip().split('\t')
                    ref_dict[tmp_pair[0]] = tmp_pair[1]
        return ref_dict
    return None

dataset_ = 'political_small'
model_ = 'toward_12'

ori_path = '{}/data/{}/{}/origin.text'.format(working_dir_, dataset_, model_)
trans_path = '{}/data/{}/{}/trans.text'.format(working_dir_, dataset_, model_)

refs = []
hyps = []
with open(ori_path, 'r') as fori:
    for line in fori.readlines():
        refs.append(line.strip())
        
with open(trans_path, 'r') as ftrans:
    for line in ftrans.readlines():
        hyps.append(line.strip())

refs = np.expand_dims(refs, axis=1)
bleu = tx.evals.corpus_bleu_moses(refs, hyps)
print('bleu w.r.t. original corpus of {} in {}: {}'.format(model_, dataset_, bleu))

ref_dict = get_human_ref_dict(dataset_)
if ref_dict:
    ref_human = []
    for _ref in refs:
        ref_human.append(ref_dict[_ref[0]])
    ref_human = np.expand_dims(ref_human, axis=1)
    bleu_human = tx.evals.corpus_bleu_moses(ref_human, hyps)
    print('bleu w.r.t. human rewritten corpus of {} in {}: {}'.format(model_, dataset_, bleu_human))

stop=1












