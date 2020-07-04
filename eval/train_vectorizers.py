from style_lexicon import load_lexicon
from utils import load_dataset
from content_preservation import mask_style_words
from content_preservation import train_word2vec_model

for dataset in ['political', 'title']:
    print('processing dataset {}...'.format(dataset))
    datatype = 'sentiment' if dataset =='yelp' else dataset

    # load style_lexicon
    styles = {0: 'binary {}'.format(datatype)}
    style_features_and_weights_path = 'style_lexicon/style_words_and_weights_{}.json'.format(dataset)
    loaded_style_lexicon = load_lexicon(styles, style_features_and_weights_path)

    all_texts = load_dataset('eval_data/{}/{}.all'.format(dataset, datatype))
    all_texts_style_masked = mask_style_words(all_texts, style_tokens=loaded_style_lexicon, mask_style=True)

    w2v_model_path = 'eval_models/word2vec_unmasked_{}'.format(dataset)
    w2v_model_style_masked_path = 'eval_models/word2vec_masked_{}'.format(dataset)

    train_word2vec_model(all_texts, w2v_model_path)
    train_word2vec_model(all_texts_style_masked, w2v_model_style_masked_path)
