import json 
import pdb
from tqdm import tqdm
import numpy as np
import argparse
import math
from classifier.clas_test_distr import generate_style_distr
from style_transfer_intensity import load_style_distributions
from style_transfer_intensity import calculate_direction_corrected_emd
from style_lexicon import load_lexicon
from utils import load_dataset, merge_datasets
from content_preservation import mask_style_words
from content_preservation import generate_style_modified_texts
from content_preservation import load_word2vec_model
from content_preservation import calculate_wmd_scores

args = argparse.ArgumentParser(description='evaluating the model')
args.add_argument('--dataset', type=str, default='yelp', help='the dataset to use')
args.add_argument('--model', type=str, default='GTAE-alfa-20200702-0', help='the model to evaluate')
args.add_argument('--eval', type=str, default='all', help='style_transfer, or content_preservation, or naturalness')
args = args.parse_args()

dataset = args.dataset
model = args.model
eval_sti, eval_cp, eval_nat = False, False, False
if args.eval == 'all':
    eval_sti, eval_cp, eval_nat = True, True, True
elif args.eval == 'style_transfer': 
    eval_sti = True
elif args.eval == 'content_preservation':
    eval_cp = True
elif args.eval == 'naturalness':
    eval_nat = True

print('========================================')
print('(1) Dataset: {} (2) Model: {}'.format(dataset, model))
print('========================================')
if eval_sti:
    print('Evaluating Style Transfer Intensity')
    print('========================================')
    # Generate ori/trans_distribution.npz
    for style_ in ['trans', 'ori']:
        print('Generating {}_distribution.npz...'.format(style_))
        test_accu = generate_style_distr(dataset, model, style_)
        if style_ == 'trans': trans_accu = test_accu

    # Calculate direction-corrected EMD
    trans_distr, trans_labels = load_style_distributions('eval_results/{}/{}/trans_distribution.npz'.format(dataset, model))
    ori_distr, ori_labels = load_style_distributions('eval_results/{}/{}/ori_distribution.npz'.format(dataset, model))
    textcnn_itensities = []
    print('Calculating direction-corrected EMD scores...')
    for i in tqdm(range(len(trans_labels))):
        tmp_textcnn_itensity = calculate_direction_corrected_emd(ori_distr[i], trans_distr[i], trans_labels[i])
        textcnn_itensities.append(tmp_textcnn_itensity)
    mean_EMD = np.mean(textcnn_itensities)
    print('transfer accuracy: {}'.format(trans_accu))
    print('mean EMD: {}'.format(mean_EMD))

if eval_cp:
    print('Evaluating Content Preservation')
    print('========================================')
    datatype = 'sentiment' if dataset == 'yelp' else dataset
    styles = {0: 'binary {}'.format(datatype)}
    style_features_and_weights_path = 'style_lexicon/style_words_and_weights_{}.json'.format(dataset)
    loaded_style_lexicon = load_lexicon(styles, style_features_and_weights_path)
    
    ori_texts = load_dataset('eval_results/{}/{}/ori.text'.format(dataset, model))
    trans_texts = load_dataset('eval_results/{}/{}/trans.text'.format(dataset, model))
    _, _, ori_texts_masked = generate_style_modified_texts(ori_texts, loaded_style_lexicon)
    _, _, trans_texts_masked = generate_style_modified_texts(trans_texts, loaded_style_lexicon)
    
    w2v_model_masked = load_word2vec_model('eval_models/word2vec_masked_{}'.format(dataset)) 
    
    wmd_scores_masked = calculate_wmd_scores(ori_texts_masked, trans_texts_masked, w2v_model_masked)
    all_wmd_scores_masked = 0
    num_wmd_scores_masked = 0
    for score_ in wmd_scores_masked:
        if not math.isinf(score_):
            all_wmd_scores_masked += score_
            num_wmd_scores_masked += 1
    mean_wmd_scores_masked = all_wmd_scores_masked / num_wmd_scores_masked
    print('mean masked WMD: {}'.format(mean_wmd_scores_masked))
    
    
if eval_nat:
    print('Evaluating Naturalness')
    print('========================================')

