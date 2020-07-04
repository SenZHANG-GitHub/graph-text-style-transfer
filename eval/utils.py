from globals import *
from scipy.stats import linregress
from sklearn.externals import joblib
import json
import math
import numpy as np
import pandas


## I/O / LOADING
def merge_datasets(dataset1, dataset2):
    x = []
    x.extend(dataset1)
    x.extend(dataset2)
    return x

def compile_binary_dataset(negative_samples, positive_samples):
    x = merge_datasets(negative_samples, positive_samples)
    y = np.concatenate([np.zeros(len(negative_samples)), np.ones(len(positive_samples))])
    return x, y

def load_dataset(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(line.strip())
    return data

def load_all_set(neg_path, pos_path):
    """e.g.
    neg_path: 'eval_data/yelp/sentiment.all.0'
    pos_path: 'eval_data/yelp/sentiment.all.1'
    DATA_VECTORIZER_PATH = './eval_models/vectorizer.pkl'
    """
    neg_data = load_dataset(neg_path)
    pos_data = load_dataset(pos_path)
    x_test, y_test = compile_binary_dataset(neg_data, pos_data)
    return x_test, y_test


def load_test_set(neg_path, pos_path):
    """e.g.
    neg_path: '../data/sentiment.test.0'
    pos_path: '../data/sentiment.test.1'
    """
    neg_yelp = load_dataset(neg_path)
    pos_yelp = load_dataset(pos_path)
    yelp_x_test, yelp_y_test = compile_binary_dataset(neg_yelp, pos_yelp)
    return yelp_x_test, yelp_y_test

def load_train_set(neg_path, pos_path):
    """ e.g.
    neg_path: '../data/sentiment.train.0'
    pos_path: '../data/sentiment.train.1'
    """
    neg_yelp = load_dataset(neg_path)
    pos_yelp = load_dataset(pos_path)
    yelp_x_train, yelp_y_train = compile_binary_dataset(neg_yelp, pos_yelp)
    return yelp_x_train, yelp_y_train

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def load_model(path):
    return joblib.load(path)

def load_turk_scores(aspect, model, param, param_val, npy_file=True):
    filetype = 'npy' if npy_file else 'npz'
    return np.load('../evaluations/human/{aspect}/{model}_{param}_{param_val}.{filetype}'.format(aspect=aspect, model=model, param=param, param_val=param_val, filetype=filetype))

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def save_model(model, path):
    joblib.dump(model, path)


## CORRELATION TESTING
def calculate_std_err_of_r(r, n):
    # find standard error of correlation coefficient (based on jstor.org/stable/2277400)
    return (1-r**2)/math.sqrt(n)

def get_margin_of_error(std_err):
    # represent one standard deviation 
    # can be used with respect to mean of data to find confidence interval
    return 1.96 * std_err

def calculate_correlations(metrics_dict, turk_scores):
    correlation_dict = {}
    number_of_samples = len(turk_scores)
    
    for metric in metrics_dict:
        automated_scores = metrics_dict[metric]
        _, _, pearson_corr, pearson_p_val, _ = linregress(automated_scores, turk_scores)
        std_error_of_r = calculate_std_err_of_r(pearson_corr, number_of_samples) 
        sample_based_margin_of_error = get_margin_of_error(std_error_of_r)
        assert pearson_p_val < 0.05

        correlation_dict[metric] = {
            'r-val': pearson_corr,
            'error_bound': sample_based_margin_of_error
        }

    return pandas.DataFrame(data=correlation_dict).transpose()


## MISCELLANEOUS
def get_val_as_str(val):
    return str(val).replace('.', '_')

def invert_dict(dictionary):
    return dict(zip(dictionary.values(), dictionary.keys()))

def calculate_corpus_level_scores(model, aspect):    
    '''
    This is how scores under "../evaluations/automated/<aspect>/corpus_level/" were obtained:
    take the mean of the sentence-level scores for a given model and aspect of evaluation. 
    
    Sentence-level scores are from the automated metrics most strongly correlated
    or in agreement with human judgments, as determined empirically.

    Parameters
    ----------
    model : str
        One of three style transfer models used in experiments (see globals.py)
    aspect : str
        One of three key aspects of style transfer model evaluation (see globals.py)
        
    Returns
    -------
    corpus_level_scores : dict
        Mapping from model hyperparameter to corpus-level score

    '''
    
    param_name = MODEL_TO_PARAM_NAMES[model]
    param_values = MODEL_TO_PARAMS[model]
    preferred_metric = PREFERRED_AUTOMATED_METRICS[aspect]

    automated_sentence_level_scores_base_path = '../evaluations/automated/{aspect}/sentence_level'.format(aspect=aspect)

    if aspect == 'style_transfer_intensity':
        automated_sentence_level_scores_path += '/scores_based_on_emd'

    corpus_level_scores = {}

    for val in param_values:
        string_val = get_val_as_str(val)
        sentence_level_scores_path = '{automated_sentence_level_scores_base_path}/{model}_{param_name}_{string_val}.npz'.format(automated_sentence_level_scores_base_path=automated_sentence_level_scores_base_path, model=model, param_name=param_name, string_val=string_val)
        npz = np.load(sentence_level_scores_path)
        
        if aspect == 'content_preservation':
            sentence_level_scores = npz[STYLE_MODIFICATION_SETTING].item()

        sentence_level_scores = sentence_level_scores[preferred_metric] 
        corpus_level_scores[val] = np.mean(sentence_level_scores) 
        
    return corpus_level_scores