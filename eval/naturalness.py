"""EVALUATION OF NATURALNESS

This code can be used to evaluate the naturalness of output sentiment texts of examined style transfer models.

For a baseline understanding of what is considered "natural," any method used for automated evaluation of naturalness 
also requires an understanding of the human-sourced input texts. 

Inspired by the adversarial evaluation approach in "Generating Sentences from a Continuous Space"
(Bowman et al., 2016), we trained unigram logistic regression classifiers and LSTM logistic regression classifiers 
on samples of input texts and output texts for each style transfer model.

Via adversarial evaluation, the classifiers must distinguish human-generated inputs from machine-generated outputs. 
The more natural an output is, the likelier it is to fool an adversarial classifier.

We calculate percent agreement with human judgments. Both classifiers show greater agreement on which texts are
considered more natural with humans given relative scoring tasks than with those given absolute scoring tasks.

Usage:
    - View percent agreement between automated scores and human scores              -> display_agreements()
    - Calculate naturalness scores for texts with clf, a NaturalnessClassifier      -> clf.score(...)

You can find examples of more detailed usage commands below.

"""

from collections import Counter
from globals import *
from keras.models import load_model as load_keras_model
from keras.preprocessing.sequence import pad_sequences
from tokenizer import RE_PATTERN
from utils import get_val_as_str, invert_dict, load_dataset, load_model, load_turk_scores, merge_datasets
import numpy as np
import pandas as pd
import re

ASPECT = 'naturalness'
# AUTOMATED_EVALUATION_BASE_PATH = '../evaluations/automated/{ASPECT}/sentence_level'.format(ASPECT=ASPECT)
CLASSIFIER_BASE_PATH = 'eval_models/naturalness_classifiers'
MAX_SEQ_LEN = 30 # for neural classifier 
TEXT_VECTORIZER = load_model('eval_models/vectorizer.pkl')

# adjust vocabulary to account for unknowns
VOCABULARY = TEXT_VECTORIZER.vocabulary_
INVERSE_VOCABULARY = invert_dict(VOCABULARY)
VOCABULARY[INVERSE_VOCABULARY[0]] = len(VOCABULARY)
VOCABULARY['CUSTOM_UNKNOWN'] = len(VOCABULARY)+1

## DATA PREP
def convert_to_indices(text):
    # tokenize input text
    tokens = re.compile(RE_PATTERN).split(text)    
    non_empty_tokens = list(filter(lambda token: token, tokens))
    
    indices = []
    
    # collect indices of tokens in vocabulary
    for token in non_empty_tokens:
        if token in VOCABULARY:
            index = VOCABULARY[token]
        else:
            index = VOCABULARY['CUSTOM_UNKNOWN']
            
        indices.append(index)
    
    return indices

def format_inputs(texts):
    # prepare texts for use in neural classifier
    texts_as_indices = []
    for text in texts:
        texts_as_indices.append(convert_to_indices(text))
    return pad_sequences(texts_as_indices, maxlen=MAX_SEQ_LEN, padding='post', truncating='post', value=0.)


## NATURALNESS CLASSIFIERS
class NaturalnessClassifier: 
    '''
    An external classifier was trained for each examined style transfer model - 
    more specifically using its inputs and outputs, of course excluding test samples.
    
    Use UnigramBasedClassifier or NeuralBasedClassifier to load a 
    trained classifier and score texts of a given style transfer model.
    The scores represent the probabilities of the texts being 'natural'.
    
    '''
    
    pass

class UnigramBasedClassifier(NaturalnessClassifier):
    def __init__(self, style_transfer_model_name, text_vectorizer=TEXT_VECTORIZER):
        self.path = '{CLASSIFIER_BASE_PATH}/unigram_{style_transfer_model_name}.pkl'.format(CLASSIFIER_BASE_PATH=CLASSIFIER_BASE_PATH, style_transfer_model_name=style_transfer_model_name)
        self.classifier = load_model(self.path)
        self.text_vectorizer = text_vectorizer
        
    def score(self, texts):
        vectorized_texts = self.text_vectorizer.transform(texts)
        distribution = self.classifier.predict_proba(vectorized_texts)
        scores = distribution[:,1] # column 1 represents probability of being 'natural'
        return scores

class NeuralBasedClassifier(NaturalnessClassifier):
    def __init__(self, style_transfer_model_name):
        self.path = '{CLASSIFIER_BASE_PATH}/neural_{style_transfer_model_name}.h5'.format(CLASSIFIER_BASE_PATH=CLASSIFIER_BASE_PATH, style_transfer_model_name=style_transfer_model_name)
        self.classifier = load_keras_model(self.path)

    def score(self, texts):
        inps = format_inputs(texts)
        distribution = self.classifier.predict(inps)
        scores = distribution.squeeze()
        return scores
    
    
## CALCULATION OF AGREEMENTS
def generate_judgments(input_text_scores, output_text_scores):
    '''
    Compare naturalness scores of input and output texts, representing
    the case where an input is scored as more natural with 1, output with 0,
    and neither with None. Generate "judgments" with such labels.

    Parameters
    ----------
    input_text_scores : numpy.ndarray
        Naturalness scores assigned to input texts
    output_text_scores : numpy.ndarray
        Naturalness scores assigned to output texts
        
    Returns
    -------
    judgments : numpy.ndarray
        Labels representing which texts were marked as more natural

    '''
        
    judgments = []
    
    for i in range(len(input_text_scores)):
        input_text_score = input_text_scores[i]
        output_text_score = output_text_scores[i]
        
        if input_text_score != output_text_score:
            # represent input text being scored as more natural as 1, otherwise 0
            val = int(input_text_score > output_text_score)
        else:
            val = None
        judgments.append(val)
        
    return np.array(judgments)

def format_relative_judgments(judgments):
    '''
    Raters provided judgments of which of a given input ('A') and
    output text ('B') is more natural. Represent 'A' as 1 and 'B' as 0
    for downstream comparison with judgments from other scoring methods
    that use this representation.

    Parameters
    ----------
    judgments : list 
        List of 'A', 'B', and/or None judgments
        
    Returns
    -------
    List of formatted judgments

    '''
    
    judgments_map = {'A': 1, 'B': 0, None: None}
    return list(map(lambda judgment: judgments_map[judgment], judgments))

# def display_agreements():
#     '''
#     Display percent agreements of automated naturalness metrics (scores from adversarial classifiers) 
#     with human evaluation scores for examined style transfer models.
    
#     Percent agreement is the percent of samples where humans and a classifier both rate a text 
#     as more natural than the other. 

#     '''
    
#     # load average human judgments for absolute scoring task of input texts
#     # (relative scoring task depends on both input and output texts - those judgments are loaded below, per style transfer model) 
#     human_absolute_judgments_for_input_texts = np.load('../evaluations/human/{ASPECT}/input_texts.npz'.format(ASPECT=ASPECT))['absolute']

#     for model in MODEL_TO_PARAMS:
#         print('{model} AGREEMENTS:'.format(model=model))
#         print('====================')
#         param_name = MODEL_TO_PARAM_NAMES[model]
#         param_values = MODEL_TO_PARAMS[model]
        
#         # load automated scores for input texts
#         automated_scores_for_input_texts = np.load('{AUTOMATED_EVALUATION_BASE_PATH}/input_texts.npz'.format(AUTOMATED_EVALUATION_BASE_PATH))
#         unigram_scores_for_input_texts = automated_scores_for_input_texts['unigram'].item()[model]
#         neural_scores_for_input_texts = automated_scores_for_input_texts['neural'].item()[model] 

#         # track number of agreements with human judgments
#         agreements = {
#             'unigram_and_absolute': 0,
#             'unigram_and_relative': 0,
#             'neural_and_absolute' : 0,
#             'neural_and_relative' : 0
#         }

#         total_number_of_samples = 0
        
#         for val in param_values:
#             # limited human evaluations to only output text of gamma=15 for DAR model, due to high costs of annotation
#             if model == 'DAR' and val in [0.1, 1, 500]:
#                 continue

#             str_val = get_val_as_str(val)

#             # load human judgments of output texts (majority-based for relative scoring tasks; average for absolute scoring tasks)
#             human_judgments = load_turk_scores(ASPECT, model, param_name, str_val, npy_file=False)
#             human_absolute_judgments_for_output_texts = human_judgments['absolute'] 
#             human_absolute_judgments = generate_judgments(human_absolute_judgments_for_input_texts, human_absolute_judgments_for_output_texts)
#             human_relative_judgments = format_relative_judgments(human_judgments['relative']) 
            
#             # load classifier scores of output texts
#             automated_scores_for_output_texts_path = '{AUTOMATED_EVALUATION_BASE_PATH}/{model}_{param_name}_{str_val}.npz'.format(AUTOMATED_EVALUATION_BASE_PATH=AUTOMATED_EVALUATION_BASE_PATH, model=model, param_name=param_name, str_val=str_val)
#             automated_scores_for_output_texts = np.load(automated_scores_for_output_texts_path) 
            
#             # count number of agreements between what classifiers mark as more natural and what humans mark as more natural
#             unigram_scores_for_output_texts = automated_scores_for_output_texts['unigram']
#             unigram_judgments = generate_judgments(unigram_scores_for_input_texts, unigram_scores_for_output_texts)
#             agreements['unigram_and_absolute'] += np.sum(unigram_judgments == human_absolute_judgments)
#             agreements['unigram_and_relative'] += np.sum(unigram_judgments == human_relative_judgments)

#             neural_scores_for_output_texts = automated_scores_for_output_texts['neural']
#             neural_judgments = generate_judgments(neural_scores_for_input_texts, neural_scores_for_output_texts)
#             agreements['neural_and_absolute'] += np.sum(neural_judgments == human_absolute_judgments)
#             agreements['neural_and_relative'] += np.sum(neural_judgments == human_relative_judgments)
            
#             number_of_samples = len(neural_judgments)
#             total_number_of_samples += number_of_samples

#         # get percent agreements based on counts aggregated over multiple params
#         for a in agreements:
#             agreement_ratio = round(agreements[a] / total_number_of_samples * 100, 2)
#             print("{a.upper()}: {agreement_ratio}%".format(agreement_ratio=agreement_ratio))
#         print()
        
        
# EXAMPLE USAGE (uncomment the following to get naturalness scores with a trained adversarial classifier)

# model = 'CAAE'
# param = MODEL_TO_PARAM_NAMES[model]
# val = get_val_as_str(0.1)

# # load data
# negative_to_positive_transfers = load_dataset(f'../transfer_model_outputs/{model}/{param}_{val}/sentiment.test.0.tsf')
# positive_to_negative_transfers = load_dataset(f'../transfer_model_outputs/{model}/{param}_{val}/sentiment.test.1.tsf')
# output_texts = merge_datasets(negative_to_positive_transfers, positive_to_negative_transfers)

# # score
# neural_classifier = NeuralBasedClassifier(model)
# print(neural_classifier.score(output_texts))


# def toward_and_our_model_naturalness(sentence_path, model):

#     reference_sentence = []
#     candidates_sentence = []

#     with open(sentence_path,'r') as mixed_sentence:
#         for i,line in enumerate(mixed_sentence):
#             if i % 2==0:
#                 reference_sentence.append(line.split("\n")[0])
#             else:
#                 candidates_sentence.append(line.split("\n")[0])

#     neural_classifier = NeuralBasedClassifier(model)
#     scores = neural_classifier.score(candidates_sentence)

#     return np.array(scores).mean()


# if __name__ == "__main__":
#     sentence_path = '../evaluation_results/Yelp/val.12'

#     score_ARAE = toward_and_our_model_naturalness(sentence_path, "ARAE")
#     score_CAAE = toward_and_our_model_naturalness(sentence_path, "CAAE")
#     score_DAR = toward_and_our_model_naturalness(sentence_path, "DAR")
