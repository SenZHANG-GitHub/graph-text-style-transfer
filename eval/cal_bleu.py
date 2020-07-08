import os
import argparse
from utils import MessageContainer
import texar as tx 
import numpy as np
import pdb

args = argparse.ArgumentParser(description='evaluating the model')
args.add_argument('--dataset', type=str, default='yelp', help='the dataset to use')
args.add_argument('--models', type=str, default='CAAE,ARAE,DAR,DAR_DeleteOnly,DAR_RetrieveOnly', help='the model to evaluate')
# args.add_argument('--model', type=str, default='GTAE-alfa-20200702-0,StyleTransformer-multi,DualRL,StyleTransformer-cond,UnsuperMT_Zhang,UnpairedRL_Xu,TemplateBase_Li,DeleteOnly_Li,DeleteRetrieve_Li,ca_IMaT,IMaT,multi_decoder_IMaT,Seq2SentiSeq-7,Seq2SentiSeq-9,Seq2SentiSeq-5,Seq2SentiSeq-3,Seq2SentiSeq-1,BackTranslation_Pr,style_emb_IMaT,RetrieveOnly_Li,ARAE', help='the model to evaluate')
# args.add_argument('--model', type=str, default='dar_IMaT,StyleEmbedding_Fu,Multidecoder_Fu', help='the model to evaluate')
args = args.parse_args()

# def get_human_ref_dict(dataset_):
#     ref_dict = dict()
#     if dataset_ == "yelp_small":
#         for _i in [0,1]:
#             with open('{}/data/yelp_small/reference-human.{}'.format(working_dir_, _i), 'r') as ftmp:
#                 for line in ftmp.readlines():
#                     tmp_pair = line.strip().split('\t')
#                     ref_dict[tmp_pair[0]] = tmp_pair[1]
#         return ref_dict
#     return None

dataset_ = args.dataset
models = args.models.split(',')

for model_ in models:
    ori_path = 'eval_results/{}/{}/ori.text'.format(dataset_, model_)
    trans_path = 'eval_results/{}/{}/trans.text'.format(dataset_, model_)

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
    print('bleu score of {}: {}'.format(model_, bleu))

# ref_dict = get_human_ref_dict(dataset_)
# if ref_dict:
#     ref_human = []
#     for _ref in refs:
#         ref_human.append(ref_dict[_ref[0]])
#     ref_human = np.expand_dims(ref_human, axis=1)
#     bleu_human = tx.evals.corpus_bleu_moses(ref_human, hyps)
#     print('bleu w.r.t. human rewritten corpus of {} in {}: {}'.format(model_, dataset_, bleu_human))

# stop=1












