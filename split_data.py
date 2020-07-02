import pdb
from tqdm import tqdm

all_lines = []
curr_ind = 0
interval = 80000

def write_lines(lines, ind):
    with open('adjs/sentiment.train.text-{}'.format(ind), mode='w') as fw:
        for line in lines:
            fw.write(line)

with open('data/yelp/sentiment.train.text', mode='r') as f:
    for cnt, line in tqdm(enumerate(f.readlines())):
        if cnt % interval == 0 and cnt > 0:
            write_lines(all_lines, curr_ind)
            curr_ind += 1
            all_lines = []
        all_lines.append(line)
    write_lines(all_lines, curr_ind)
