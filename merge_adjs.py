from tqdm import tqdm
with open('adjs/sentiment.train.adjs', mode='w') as fw:
    for i_adj in range(6):
        with open('adjs/sentiment.train.adjs-{}'.format(i_adj), mode='r') as fr:
            print('processing adjs/sentiment.train.adjs-{}...'.format(i_adj))
            for line in tqdm(fr.readlines()):
                fw.write(line)