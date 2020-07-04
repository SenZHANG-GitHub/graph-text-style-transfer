"""GENERATION OF STYLE LEXICON 

The following steps were used to automatically generate a sentiment style lexicon:
    1. Load a text dataset, with style labels.
    2. Train a logistic regression model on the labeled data (train(...))
    3. Extract nonzero weights and corresponding features from the model,
        using an experimentally determined threshold on those weights.
        For details on the threshold, please see select_feature_numbers(...).
        
To load the lexicon, use load_lexicon().

"""

from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from tokenizer import tokenize
from utils import invert_dict, load_json, load_all_set, load_model, save_json, save_model
import numpy as np
import pdb
import os
import shutil

def fit_vectorizer(inp, data_vectorizer_path):
    '''
    Fit a vectorizer that can transform text inputs into flat one-hot encodings
    using custom tokenization rules (see tokenizer.py).

    Parameters
    ----------
    inp : list
        Text inputs  

    Returns
    -------
    vectorizer : sklearn.feature_extraction.text.CountVectorizer
        Fitted vectorizer

    '''
    
    vectorizer = CountVectorizer(binary=True, tokenizer=tokenize)
    vectorizer.fit(inp)
    save_model(vectorizer, data_vectorizer_path)
    return vectorizer

def train(regularization_type, C, vec_x_train, y_train):    
    '''
    Train a logistic regression model for obtaining style-related weights of a dataset.

    Parameters
    ----------
    regularization_type : str
        "l1" or "l2" regularization
    C : float
        Inverse of regularization strength
    vec_x_train : scipy.sparse.csr.csr_matrix
        One-hot feature encodings
    y_train : numpy.ndarray
        Style labels

    Returns
    -------
    lr : sklearn.linear_model.LogisticRegression
        Trained model

    '''
    
    lr = LogisticRegression(penalty=regularization_type, C=C)
    lr.fit(vec_x_train, y_train)  
    return lr

def extract_nonzero_weights(model):
    '''
    Extract nonzero weights of model, and corresponding features.

    Parameters
    ----------
    model : sklearn.linear_model.LogisticRegression
        Trained model

    Returns
    -------
    nonzero_weights : numpy.ndarray
        Extracted weights
    feature_numbers_for_nonzero_weights : numpy.ndarray 
        Feature numbers

    '''
    
    all_weights = model.coef_
    
    feature_numbers_for_nonzero_weights = []
    nonzero_weights = []
    
    for style_number, weights in enumerate(all_weights):
        # ignore features with nonzero weights (no impact on style labeling)
        feature_numbers = np.where(abs(weights) > 0.0)[0]
        feature_numbers_for_nonzero_weights.append(feature_numbers)
        nonzero_weights.append(weights[feature_numbers])
    
    return np.array(nonzero_weights), np.array(feature_numbers_for_nonzero_weights)

def select_feature_numbers(weights, number_of_standard_deviations=2):
    '''
    Select features of a model whose corresponding weights are at or beyond 
    a given number of standard deviations away from the mean weight.
    
    Weights further from the mean signify greater impact of the corresponding  
    features on the outcome of style labels. That is, the selected features  
    are fewer in number, but more strongly weighted.
    
    Parameters
    ----------
    weights : numpy.ndarray
        Model weights
    number_of_standard_deviations : int
        Number of standard deviations away from the mean weight

    Returns
    -------
    feature_numbers : numpy.ndarray
        Selected feature numbers
    
    ''' 
    
    standard_deviation = np.std(weights)
    mean = np.mean(weights)
    left_bound = mean - number_of_standard_deviations * standard_deviation
    right_bound = mean + number_of_standard_deviations * standard_deviation
    feature_numbers = np.where((weights < left_bound) | (weights > right_bound))[0]
    return feature_numbers    

def extract_ranked_features(nonzero_weights, feature_numbers, inverse_vocabulary, weighted_feature_numbers):
    '''
    Obtain features (tokens) corresponding to given feature numbers.

    Parameters
    ----------
    nonzero_weights : numpy.ndarray
        Model weights for a given style
    feature_numbers : numpy.ndarray
        Select feature numbers from entire vocabulary, based on weight thresholding
    inverse_vocabulary : dict
        Mapping of feature number to feature
    weighted_feature_numbers : numpy.ndarray
        Select feature numbers with respect to list of weighted feature numbers (not entire feature set)

    Returns
    -------
    Mapping of feature to weight, ranked by weight
    
    '''
    
    dictionary = {} 

    for index, feature_number in enumerate(weighted_feature_numbers):
        feature = inverse_vocabulary[feature_number]
        weight = nonzero_weights[feature_numbers[index]]
        dictionary[feature] = weight

    return sorted(dictionary.items(), key=itemgetter(1))
        
def collect_style_features_and_weights(weights, styles, inverse_vocabulary, feature_numbers, number_of_standard_deviations=2):
    '''
    Collect style features per style class, based on features whose weights 
    are selected using a threshold of standard deviations away from the mean weight. 
        
    Given the threshold, there is a tradeoff of capturing more style features and reducing noise
    in the lexicon. At the expense of not capturing some style features, we opt for a smaller 
    default threshold to reduce noise and minimize the risk of removing content features. 
    This is critical to downstream evaluations of content preservation.  
        
    Parameters
    ----------
    weights : numpy.ndarray
        Select weights
    styles : dict
        Mapping of style indices to style names
    inverse_vocabulary : dict
        Mapping of feature number to feature
    feature_numbers : numpy.ndarray
        Feature numbers corresponding to select weights
    number_of_standard_deviations : int
        See select_feature_numbers.py for description

    Returns
    -------
    style_features_and_weights : dict
        Mapping of style name to style features and weights
    
    '''
    
    style_features_and_weights = {}
    
    for style_number, style_weights in enumerate(weights):
        style = styles[style_number]
        selected_feature_numbers = select_feature_numbers(style_weights, number_of_standard_deviations) 
        weighted_feature_numbers = feature_numbers[style_number][selected_feature_numbers]        
        ranked_features = extract_ranked_features(style_weights, selected_feature_numbers, inverse_vocabulary, weighted_feature_numbers)    
        style_features_and_weights[style] = ranked_features
        
    return style_features_and_weights    

def load_lexicon(styles, style_features_and_weights_path):
    # collect style words from existing set of style features and weights
    style = styles[0]
    style_features_and_weights = load_json(style_features_and_weights_path)
    return set(map(lambda x: x[0], style_features_and_weights[style]))

if __name__ == '__main__':
    data_info = {
        'yelp': 'sentiment',
        'political': 'political',
        'title': 'title'
    }
    
    if not os.path.isdir('eval_models'): os.mkdir('eval_models')
    if not os.path.isdir('style_lexicon'): os.mkdir('style_lexicon')
    
    for dataset, prefix in data_info.items():
        print('========================================')
        print('Processing {} dataset...'.format(dataset))
        print('========================================')
        ## STEP 1.
        print('STEP 1: LOAD AND PREPARE DATA...')
        neg_path = 'eval_data/{}/{}.all.0'.format(dataset, prefix)
        pos_path = 'eval_data/{}/{}.all.1'.format(dataset, prefix)
        x_tr, y_tr = load_all_set(neg_path, pos_path) 

        # edit fit_vectorizer() to create custom vectorizer for dataset
        # otherwise, load existing vectorizer under data_vectorizer_path
        data_vectorizer_path = 'eval_models/vectorizer_{}.pkl'.format(dataset)
        if os.path.isfile(data_vectorizer_path):
            os.remove(data_vectorizer_path)
        fit_vectorizer(x_tr, data_vectorizer_path)
        vectorizer = load_model(data_vectorizer_path)
        inverse_vocabulary = invert_dict(vectorizer.vocabulary_)

        ## STEP 2.
        print('STEP 2: TRAIN MODEL TO OBTAIN STYLE WEIGHTS...')
        # to experiment with style weighting, edit parameters and train new model
        regularization_type = 'l1'
        C = 3
        lr_path = "eval_models/style_weights_extractor_{regularization_type}_reg_C_{C}_{dataset}.pkl".format(regularization_type=regularization_type, C=C, dataset=dataset)
        if os.path.isfile(lr_path):
            os.remove(lr_path)
        vec_x_tr = vectorizer.transform(x_tr)
        lr_model = train(regularization_type, C, vec_x_tr, y_tr)
        save_model(lr_model, lr_path)
        model = load_model(lr_path)

        ## STEP 3.
        print('STEP 3: EXTRACT STYLE FEATURES AND WEIGHTS...')
        # if using another type of style, adjust the set of styles below
        styles = {0: 'binary {}'.format(prefix)}
        style_features_and_weights_path = 'style_lexicon/style_words_and_weights_{}.json'.format(dataset)
        if os.path.isfile(style_features_and_weights_path):
            os.remove(style_features_and_weights_path)
        nonzero_weights, feature_numbers = extract_nonzero_weights(model) 
        style_features_and_weights = collect_style_features_and_weights(nonzero_weights, styles, inverse_vocabulary, feature_numbers)
        save_json(style_features_and_weights, style_features_and_weights_path)
        