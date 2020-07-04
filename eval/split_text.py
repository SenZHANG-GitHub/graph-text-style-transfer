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

def split_GTAE(dataset, model, filename):
    """
    dataset: 'yelp'
    model: 'GTAE-alfa-20200702-0'
    filename: 'val.13'
    """
    datatype = 'sentiment' if dataset == 'yelp' else dataset
    filepath = 'eval_results/{}/{}'.format(dataset, model)
    all_text = []
    with open('{}/{}'.format(filepath, filename), mode='r') as f:
        for line in f.readlines():
            all_text.append(line)
    ori_text = []
    trans_text = []
    
    for i in range(len(all_text)//2):
        ori_text.append(all_text[2 * i])
        trans_text.append(all_text[2 * i + 1])
    
    # Get the labels for ori_text and trans_text
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
    for ori_text_ in ori_text:
        if ori_text_ not in raw_text_labels_dict.keys():
            pdb.set_trace()
            raise ValueError('The original text is not in the dataset')
        ori_labels.append(raw_text_labels_dict[ori_text_])
        trans_labels.append(1 - raw_text_labels_dict[ori_text_])
    
    write_text('{}/ori.text'.format(filepath), ori_text)
    write_text('{}/trans.text'.format(filepath), trans_text)
    write_label('{}/ori.label'.format(filepath), ori_labels)
    write_label('{}/trans.label'.format(filepath), trans_labels)
    

if __name__ == '__main__':
    split_GTAE(args.dataset, args.model, args.filename)
