# GTAE: Graph-Transformer based Auto-Encoders for Linguistic-Constrained Text Style Transfer 

## Benchmark
+ We also provide a benchmark at <https://github.com/ykshi/text-style-transfer-benchmark>

## Requires

Preprocessing:

+ jdk 1.8
+ stanford-corenlp (<http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip>)
+ nltk==3.5
+ tqdm

------

Training:

+ python>=3.5
+ Tensorflow-gpu==1.13
+ Texar (<https://github.com/asyml/texar>)
+ Need to add data/ folder manually

## Usage

+ generate the linguistic adjacency matrices using stanford nlp
  + go into the directory of stanford nlp
    + 'cd stanford-corenlp-full-2018-02-27'
  + start server
    + 'java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 30000 &>/dev/null'
  + extract raw adjacency file for a text file, e.g.
    + python utils_preproc/stanford_dependency.py data/yelp/sentiment.train.text data/yelp/sentiment.train.adjs 9000
    + 9000 is the server port, which should be consistent with the previous step
    + for large text file, split into multiple sub-files first and run stanford_dependency in multi-processes
  + build adjacency matrices from raw adjacency file, e.g.
    + ＇python utils_preproc/dataset_read.py data/yelp/sentiment.train.adjs data/yelp/sentiment.train_adjs.tfrecords data/yelp/sentiment.train_identities.tfrecords'

+ generate vocab of trainning data, e.g.
  + 'python get_vocab.py data/yelp/sentiment.train.text data/yelp/vocab_yelp'

+ Configure your data paths and model parameters as specified in 'config_gtt.py'

+ Training:
  + 'CUDA_VISIBLE_DEVICES=0 python main.py --config config --out output_path --lambda_t_graph 0.05 --lambda_t_sentence 0.02 --pretrain_nepochs 10 --fulltrain_nepochs 3'
  + --out is necessary
  + checkpoints/ is not saved to output_path automatically (too large). Save this folder manually if necessary, otherwise it will be erased every time we run main.py

