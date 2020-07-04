import pdb
import os
import shutil

data_names = {
    'yelp': 'sentiment',
    'political': 'political',
    'title': 'title'
}

# if os.path.isdir('eval_data'):
#     shutil.rmtree('eval_data')
# os.mkdir('eval_data')
# for key_ in data_names.keys():
#     os.mkdir('eval_data/{}'.format(key_))

for dataset, prefix in data_names.items():
    neg_list = []
    pos_list = []
    total_list = []
    for subset in ['train', 'dev', 'test']:
        tmp_list = []
        tmp_label = []
        with open('../data/{}/{}.{}.text'.format(dataset, prefix, subset), mode='r') as ft:
            for line in ft.readlines():
                tmp_list.append(line)
        with open('../data/{}/{}.{}.labels'.format(dataset, prefix, subset), mode='r') as fl:
            for line in fl.readlines():
                if line[:-1] not in ['0', '1']:
                    raise ValueError('currently only support label value 0 and 1')
                tmp_label.append(line[:-1])
        if len(tmp_list) != len(tmp_label):
            raise ValueError('number of texts and labels should be the same')
        for i_ in range(len(tmp_list)):
            total_list.append(tmp_list[i_])
            if tmp_label[i_] == '0':
                neg_list.append(tmp_list[i_])
            elif tmp_label[i_] == '1':
                pos_list.append(tmp_list[i_])
            else:
                raise ValueError('currently only support label value 0 and 1')
    with open('eval_data/{}/{}.all.0'.format(dataset, prefix), mode='w') as fn:
        for line in neg_list:
            fn.write(line)
    with open('eval_data/{}/{}.all.1'.format(dataset, prefix), mode='w') as fp:
        for line in pos_list:
            fp.write(line)
    with open('eval_data/{}/{}.all'.format(dataset, prefix), mode='w') as fo:
        for line in total_list:
            fo.write(line)