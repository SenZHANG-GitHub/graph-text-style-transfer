import os
import argparse
from utils import MessageContainer

args = argparse.ArgumentParser(description='evaluating the model')
args.add_argument('--dataset', type=str, default='yelp', help='the dataset to use')
args.add_argument('--model', type=str, default='GTAE-alfa-20200702-0', help='the model to evaluate')
args = args.parse_args()

dataset = args.dataset
model = args.model
msgs = MessageContainer()

msgs.append('========================================')
msgs.append('Evaluating Bert-Scores for Content Preservation')
msgs.append('========================================')

ori_text_path = 'eval_results/{}/{}/ori.text'.format(dataset, model)
trans_text_path = 'eval_results/{}/{}/trans.text'.format(dataset, model)
os.system('bert-score -r {} -c {} --lang en -v'.format(ori_text_path, trans_text_path))


