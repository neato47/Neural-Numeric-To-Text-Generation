# Neural-Numeric-To-Text-Generation

As the data used to create these models requires permission from the Singaporean Management University, the data will not be provided. 

There are two main scripts:
 - ED.py (CNN-LSTM and Transformer-LSTM)
 - ED_transformer.py (Transformer-Transformer)

Once data is provided, run inout_to_csv.py to generate sequence data for the models, including the summaries. One file will be generated for the summary type specified in the ED_config.json under the 'summary_types' key. This key accepts summary type codes, as follows:

SETW - standard evaluation w/ TW granularity
SESTW - standard evaluation w/ TW granularity
GE - goal evaluation
*IT - if-then pattern 
*WIT - weekday if-then pattern
*CB_description - cluster-based pattern (description of week)
*CB_cluster - cluster-based pattern (pattern summary)
ST - standard trend
EC - evaluation comparison
GC - goal comparison
GA - goal assistance
*SP - standard pattern
*DB - day-based pattern

* = there is a build_x_dataset.py that is specifically used to generate the dataset for summary type x. Use this instead of inout_to_csv.py.

These summary types (as well as the implementation) are taken from the Harris et al. paper. For now, this implementation should be used for one summary type code at a time. Once the dataset file is generated, the main scripts can be run. The hyperparameters can be tweaked in the ED_config.json (CNN-LSTM, Transformer-LSTM) and transformer_config.json (Transformer-Transformer) files for the ED.py and ED_transformer.py files, respectively. In order to use the Transformer-LSTM model, flip the 'use_transformer' flag to 'true' in the ED_config.json file. If a model's training is interrupted, it can continue training from a checkpoint by running the same script with the 'continue_train' flag set to 'true'.
