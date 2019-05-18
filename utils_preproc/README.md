# Files
1. stanford_dependency.py: 
  + 功能：对文本进行依存语法解析，并且解析的结果存储到json文件中。
  + 其中，每一个each_json有三个字段：
    + 'sent':         sentence                                              (type: string)
    + 'words':        {index: word}                                         (type: dict)
    + 'relations':    词与词之间的语法关系,每一对词之间的语法关系用tuple表示的       (type: list)
  + **注意**：原始文本需要将标点符号与单词进行空格区分！因为在代码中是按照空格对句子进行单词划分的！）

2. dataset_read.py:
  + 功能: 将dataset的adjacent信息(从json文件中可以获取)表示成一个二维matrix，并且保存到TFRecord文件!

------

# Requiements and Usage
1. jdk 1.8 required

2. Configure StanfordCoreNLP
  + download stanford-corenlp-full-2018-02-27.zip
    + 'wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip'
  + go into this directory
    + 'cd stanford-corenlp-full-2018-02-27'
  + start serverr
    + 'java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000'

3. 运行stanford_dependency.py文件
  + 'python stanford_dependency.py textfile_name jsonfile_name port'
  + port should be the same as the stanford nlp server 
  + 程序会根据Stanford依存语法工具自动解析textfile_name于jsonfile_name中，jsonfile_name格式如上。
  + 需要的时候可以手动split文件到多个文件，开多进程运行程序解析

4. 运行dataset_read.py文件
  + 'python dataset_read.py json_name tfrecord_name'
  + 程序根据第2步解析得到的jsonfile中的relations生成一个two dimension matrix并且保存到tfrecord_name中去。

------

# Getting vocab

1. Texar's yelp dataset:

+ only count tokens in train.text

  + 'test = tx.data.make_vocab("train.text", return_count=True)'

+ filter tokens with counts less than 5
  + 'test = [x for _i, x in enumerate(test[0]) if test[1][_i] >= 5]
