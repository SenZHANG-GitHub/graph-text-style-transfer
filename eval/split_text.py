import pdb
from tqdm import tqdm
import argparse

args = argparse.ArgumentParser(description='split dataset into trans/ori.text and trans/ori.label')
args.add_argument('--dataset', type=str, default='yelp', help='the dataset to use')
args.add_argument('--model', type=str, default='GTAE-alfa-20200702-0', help='the model to evaluate')
args.add_argument('--filename', type=str, default='val.13', help='to sample file to evaluate')
args = args.parse_args()

def read_dataset(filepath):
    tmplist = []
    with open(filepath, mode='r') as f:
        for line in f.readlines():
            tmplist.append(line)
    return tmplist

def write_text(filepath, text_list):
    with open(filepath, mode='w') as f:
        for text_ in text_list:
            f.write(text_)
    
def write_label(filepath, label_list):
    with open(filepath, mode='w') as f:
        for label_ in label_list:
            f.write('{}\n'.format(label_))

def replace_unk(text, vocab):
    new_text = []
    for word in text.split(' '):
        word = word.replace('\n', '')
        if word not in vocab:
            new_text.append('<UNK>')
        else:
            new_text.append(word)
    return '{}\n'.format(' '.join(new_text))

def process_text_label(dataset, ori_text, trans_text, filepath):
    datatype = 'sentiment' if dataset == 'yelp' else dataset
    raw_dev_test_text = []
    raw_dev_test_labels = []
    raw_dev_test_text.extend(read_dataset('../data/{}/{}.dev.text'.format(dataset, datatype)))
    raw_dev_test_labels.extend(read_dataset('../data/{}/{}.dev.labels'.format(dataset, datatype)))
    raw_dev_test_text.extend(read_dataset('../data/{}/{}.test.text'.format(dataset, datatype)))
    raw_dev_test_labels.extend(read_dataset('../data/{}/{}.test.labels'.format(dataset, datatype)))
    vocab = read_dataset('../data/{}/vocab_{}'.format(dataset, dataset))
    for iv, vocab_ in enumerate(vocab):
        vocab[iv] = vocab_[:-1]
    raw_text_labels_dict = dict()
    for text_, label_ in zip(raw_dev_test_text, raw_dev_test_labels):
        text_ = replace_unk(text_, vocab)
        raw_text_labels_dict[text_] = int(label_[:-1])
    
    ori_labels = []
    trans_labels = []
    cntt = 0
    for ori_text_ in ori_text:
        ori_text_
        if ori_text_ not in raw_text_labels_dict.keys():
            cntt += 1
            pdb.set_trace()
            raise ValueError('The original text is not in the dataset')
        ori_labels.append(raw_text_labels_dict[ori_text_])
        trans_labels.append(1 - raw_text_labels_dict[ori_text_])
    print('total err_num: {}'.format(cntt))
    
    write_text('{}/ori.text'.format(filepath), ori_text)
    write_text('{}/trans.text'.format(filepath), trans_text)
    write_label('{}/ori.label'.format(filepath), ori_labels)
    write_label('{}/trans.label'.format(filepath), trans_labels)


def split_GTAE(dataset, model, filename):
    """
    dataset: 'yelp'
    model: 'GTAE-alfa-20200702-0'
    filename: 'val.13'
    """
    filepath = 'eval_results/{}/{}'.format(dataset, model)
    all_text = []
    with open('{}/{}'.format(filepath, filename), mode='r') as fa:
        for line in fa.readlines():
            all_text.append(line)
    ori_text_full = dict()
    # trans_text_full = []
    
    for i in range(len(all_text)//2):
        ori_text_full[all_text[2 * i]] = all_text[2 * i + 1]
    
    # Reduce the full results to ori_common.text
    ori_text = []
    trans_text = []
    vocab = read_dataset('../data/{}/vocab_{}'.format(dataset, dataset))
    for iv, vocab_ in enumerate(vocab):
        vocab[iv] = vocab_[:-1]
    cnt = 0
    with open('eval_results/{}/ori_common.text'.format(dataset), mode='r') as fc:
        for line in fc.readlines():
            if line not in ori_text_full.keys():
                # cnt += 1
                pdb.set_trace()
                raise ValueError('The common text is not in the dataset')
            ori_text.append(line)
            trans_text.append(ori_text_full[line])
    # print('num of errs: {}'.format(cnt))
    
    # Get the labels for ori_text and trans_text and write trans/ori.text/label
    process_text_label(dataset, ori_text, trans_text, filepath)
    

def split_CAAE_ARAE_DAR(dataset, model):
    """
    dataset: 'yelp'
    model: 'CAAE'
    """
    filepath = 'eval_results/{}/{}'.format(dataset, model)
    ori_text, trans_text = [], []
    with open('eval_results/{}/{}/origin_distribution.text'.format(dataset, model)) as fo:
        for line in fo.readlines():
            ori_text.append(line.split('|')[-1][1:])
    with open('eval_results/{}/{}/trans_distribution.text'.format(dataset, model)) as ft:
        for line in ft.readlines():
            trans_text.append(line.split('|')[-1][1:])

    # Get the labels for ori_text and trans_text and write trans/ori.text/label
    process_text_label(dataset, ori_text, trans_text, filepath)

    

if __name__ == '__main__':
    if 'GTAE' in args.model:
        split_GTAE(args.dataset, args.model, args.filename)
    elif args.model in ['CAAE', 'ARAE'] or 'DAR' in args.model:
        split_CAAE_ARAE_DAR(args.dataset, args.model)
