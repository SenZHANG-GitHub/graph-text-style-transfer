"""EVALUATION OF STYLE TRANSFER INTENSITY

This code can be used for evaluation of style transfer intensity (STI) between input and output sentiment texts of a style transfer model.

Past evaluations have used the number of output texts with the desired target style.
Note that STI, however, makes use of both the input and the output texts for a more meaningful evaluation.
STI captures more information, i.e. how *much* the style changed from the input to the output.

Two output texts could exhibit the same overall target style, but with one being more pronounced than the other,
e.g. "i like this" vs. "i love this !" While past evaluations do not quantify that, STI can, 
given style distributions from a style classifier trained on labeled style datasets.

The following style classifiers were trained on ../data/sentiment.train.* and 
used in experiments to obtain style distributions:
    - fasttext (https://github.com/facebookresearch/fastText)
    - textcnn (https://github.com/DongjunLee/text-cnn-tensorflow)
   
"""

# working_dir_ = '/home/dm/Documents/text_generation/style-transfer-model-evaluation-master/classifiers'

from pyemd import emd
import numpy as np
import os
# os.chdir(working_dir_)

# dataset_ = 'title_small'
# model_ = 'GTAE-31-17'

# ori_npz_path = '{}/results/{}/{}/ori_distribution.npz'.format(working_dir_, dataset_, model_)
# trans_npz_path = '{}/results/{}/{}/trans_distribution.npz'.format(working_dir_, dataset_, model_)

## SCORING OF DIRECTION-CORRECTED EMD
def load_style_distributions(path): 
    distributions = np.load(path)
    return np.array(distributions['textcnn']).astype('float64'), np.array(distributions['label']).astype('int')

def calculate_emd(input_distribution, output_distribution):   
    '''
    Calculate Earth Mover's Distance (aka Wasserstein distance) between 
    two distributions of equal length.

    Parameters
    ----------
    input_distribution : numpy.ndarray
        Probabilities assigned to style classes for an input text
    output_distribution : numpy.ndarray
        Probabilities assigned to style classes for an output text, e.g. of a style transfer model
        
    Returns
    -------
    Earth Mover's Distance (float) between the two given style distributions

    '''
    
    N = len(input_distribution)
    distance_matrix = np.ones((N, N))
    return emd(input_distribution, output_distribution, distance_matrix)

def account_for_direction(input_target_style_probability, output_target_style_probability):
    '''
    In the context of EMD, more mass (higher probability) placed on a target style class
    in the style distribution of an output text (relative to that of the input text)
    indicates movement in the correct direction of style transfer. 
    
    Otherwise, the style transfer intensity score should be penalized, via application
    of a negative direction factor.

    Parameters
    ----------
    input_target_style_probability : float
        Probability assigned to target style in the style distribution of an input text
    output_target_style_probability : float
        Probability assigned to target style in the style distribution of an output text, e.g. of a style transfer model
        
    Returns
    -------
    1 if correct direction of style transfer, else -1

    '''
    
    if output_target_style_probability >= input_target_style_probability:
        return 1
    return -1

def calculate_direction_corrected_emd(input_distribution, output_distribution, target_style_class): 
    '''
    Calculate Earth Mover's Distance (aka Wasserstein distance) between 
    two distributions of equal length, with correction for direction.
    That is, penalize the score if the output style distribution displays
    change of style in the wrong direction, i.e. away from the target style.

    Parameters
    ----------
    input_distribution : numpy.ndarray
        Probabilities assigned to style classes for an input text
    output_distribution : numpy.ndarray
        Probabilities assigned to style classes for an output text, e.g. of a style transfer model
    target_style_class : int
        Label of the intended style class for a style transfer task
        
    Returns
    -------
    Direction-corrected Earth Mover's Distance (float) between the two given style distributions

    '''
    
    emd_score = calculate_emd(input_distribution, output_distribution)
    direction_factor = account_for_direction(input_distribution[target_style_class], output_distribution[target_style_class])
    return emd_score*direction_factor


## CALCULATION OF CORRELATIONS
def extract_scores_for_style_class(style_distributions, style_class):
    '''
    Given style distributions for a set of texts,
    extract probabilities for a given style class
    across all texts.
    
    Parameters
    ----------
    style_distributions : numpy.ndarray
        Style distributions for a set of texts
    style_class : int
        Number representing a particular style in a set of styles, 
        e.g. 0 for negative sentiment and 1 for positive sentiment
        
    Returns
    -------
    Probabilities (numpy.ndarray) for the given style class across all texts

    '''
    
    return style_distributions[:,style_class]

# def display_correlation_tables():
#     '''
#     Display correlation of automated style transfer metrics with
#     averaged human evaluations of outputs of examined style transfer models.
    
#     ''' 
    
#     # load style distributions for input texts
#     ori_text_textcnn_distr, ori_labels = load_style_distributions(ori_npz_path)
    
#     # load style distributions for output text of style transfer model
#     trans_text_textcnn_distr, trans_labels = load_style_distributions(trans_npz_path)
    
#     number_of_scores = len(trans_labels)
    
#     # collect style transfer intensities based on EMD
#     textcnn_intensities = []

#     for i in range(number_of_scores):
#         # the label of transfered style (1 - original label)
#         target_style_class = trans_labels[i]

#         # if output does not show greater probability than input for target style class, negate the EMD score
#         textcnn_intensity = calculate_direction_corrected_emd(ori_text_textcnn_distr[i], trans_text_textcnn_distr[i], target_style_class)

#         textcnn_intensities.append(textcnn_intensity)
    
#     # textcnn_intensities: [num_samples] each sentence has a EMD score
#     print('mean EMD of {} in {}: {}'.format(model_, dataset_, np.mean(textcnn_intensities)))

# if __name__ == '__main__':
#     display_correlation_tables()