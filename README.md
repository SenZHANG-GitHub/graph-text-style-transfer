# Graph Text Style Transfer

## Requires

+ python>=3.5
+ Texar
+ Tensorflow-gpu==1.13

## Usage

+ use scripts in 'utils_preproc' to preprocess the texts (get the adjacency matrices using stanford nlp)
+ check 'config_rewrite.py' and change dataset path
+ 'python main.py --config config_rewrite --out output_path' 
  + --out is necessary
  + default for --config is writen in main.py
  + checkpoints/ is not saved to output_path currently (too large). Need to save this folder manually if necessary, otherwise it will be erased every time we run main.py

## Description

+ to be added
