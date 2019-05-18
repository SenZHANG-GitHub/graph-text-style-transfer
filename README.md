# Graph Text Style Transfer

## Requires

------
Preprocessing:

+ jdk 1.8
+ stanford-corenlp (<http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip>)

------
Training:

+ python>=3.5
+ Tensorflow-gpu==1.13
+ Texar (<https://github.com/asyml/texar>)

------

## Usage

+ generate the linguistic adjacency matrices using stanford nlp
  + go into the directory of stanford nlp
    + 'cd stanford-corenlp-full-2018-02-27'
  + start server
    + 'java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 30000'
  + extract raw adjacency file for a text file
    + python utils_preproc/stanford_dependency.py textfile_name jsonfile_name 9000
    + 9000 is the server port, which should be consistent with the previous step
    + for large text file, split into multiple sub-files first and run stanford_dependency in multi-processes
  + build adjacency matrices from raw adjacency file
    + ＇python utils_preproc/dataset_read.py jsonfile_name tfrecord_name'

+ generate vocab of trainning data
  + 'python get_vocab.py train_file vocab_file'

+ Configure your data paths and model parameters as specified in 'config_gtt.py'

+ Training:
  + 'python main.py --config config_gtt --out output_path'
  + --out is necessary
  + checkpoints/ is not saved to output_path automatically (too large). Save this folder manually if necessary, otherwise it will be erased every time we run main.py

