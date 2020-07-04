# hyperparameter settings for each examined style transfer model
MODEL_TO_PARAMS = {
    'ARAE': [1, 5, 10], 
    'CAAE': [0.01, 0.1, 0.5, 1, 5], 
    'DAR' : [0.1, 1, 15, 500]
}

# names of model hyperparameters used in training
MODEL_TO_PARAM_NAMES = {
    'ARAE': 'lambda', 
    'CAAE': 'rho',
    'DAR' : 'gamma'
}


# preferred settings for automated metrics per aspect of evaluation,
# according to strongest correlation / agreement with human judgments
STYLE_MODIFICATION_SETTING = 'style_masked'

PREFERRED_AUTOMATED_METRICS = {
    'style_transfer_intensity': 'textcnn',
    'content_preservation': 'WMD',
    'naturalness': 'neural'
}