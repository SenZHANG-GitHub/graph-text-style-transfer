Dependencies: 
+ gensim, keras==2.2.4, pyemd, scipy, scikit-learn==0.20, matplotlib, pandas
+ pip install -r requirements.txt

Folder **style_lexicon** - words used to compile style lexicon used in evaluation of content preservation

Folder **classifier** - codes to get style classifier for the evaluation of style transfer 

Folder **evaluation_models** - auxiliary models used for running evaluation experiments

## Get style lexicon
Step 1: (Only Need to Do Once)
+ ```python prepare_eval_data.py```
+ Create data.all, data.all.0 and data.all.1 in Folder **eval_data** that corresponds to all, negative and positive texts
Step 2:
+ ```python style_lexicon.py```
+ Create style_weights_extract_l1_reg_C_3_dataset.pkl and vectorizer_dataset.pkl in Folder **eval_models**
+ Create style_words_and_weights_dataset.json in Folder **style_lexicon**
+ Create vecterizer_dataset.pkl in **eval_models** as well

## Get style classifer and generate distribution files for Style Transfer 
Step 1: (Only Need to Do Once)
+ Go into the Folder **classifier**
+ ```cd classifier/```

Step 2: (Only Need to Do Once)
+ Generate classifier ckpt-1 in Folder **classifier/ckpt_dataset**
+ ```python clas_train.py --config config_train_yelp --dataset yelp```
+ ```python clas_train.py --config config_train_political --dataset political```
+ ```python clas_train.py --config config_train_title --dataset title```

Step 3: (**Need to Generate trans/ori.text/label for Every Model before Evaluation!**)
+ Generate trans/ori.text, trans/ori.label from raw text (e.g. val.13 for GTAE)
+ For GTAE: ```python split_text.py --dataset yelp --model GTAE-alfa-20200702-0 --filename val.13```

Step 4:
+ ```python eval_main.py --dataset yelp --model XXX --eval style_transfer```
+ Style distributions for trans.text and ori.text are implicitly generated

## Get vectorizers for Content Preservation
Step 1: (Only Need to Do Once)
+ ```python train_vectorizers.py```
+ generate word2vec_masked/unmasked_yelp/political/title

Step 2:
+ ```python eval_main.py --dataset yelp --model XX --eval content_preservation```

## Get naturalness scores
Step 1: (Only Need to Do Once)
+ Download the trained naturalness_classifiers/ and vecterizer.pkl on https://github.com/passeul/style-transfer-model-evaluation/tree/master/models into **eval_models**

Step 2:
+ ```python eval_main.py --dataset yelp --model GTAE-alfa-XX --eval naturalness```

## Get the above metrics at the same time
Step 1: 
+ ```python eval_main.py --dataset yelp --model GTAE-alfa-XX (--eval all)```

## Get bert-score for Content Preservation
Step 1: (Only Need to Do Once)
+ Install bert-score and modifying the codes to disable warning
    + git clone https://github.com/Tiiiger/bert_score
    + cd bert_score
    + edit bert_score_cli/score.py and add the following two lines
    + ```import logging```
    + ```logging.basicConfig(level=logging.ERROR)```
    + pip install .
+ ```python eval_bert.py --dataset yelp --model GTAE-alfa-XX ```
+ ```--rescale-with-baseline``` is used in eval_bert.py

## Get BLEU scores 
Step 1: 
+ ```python cal_bleu.py --dataset yelp --models CAAE,ARAE,DAR```
+ ```--models``` instead of ```--model``` -> Allows specifying multiple models to evaluate at the same time, which should be separated by comma


