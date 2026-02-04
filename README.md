# PICARD: Phishing Intelligence Corpus for Artificial Intelligence Research and Defense: A Massive Approach to Informing Artificial Intelligence Phishing Defenses

AKA Building a dataset of AI-generated phishing messages for use in detecting and stopping AI-generated phishing attacks

## TODO:

- [x] Get unfiltered spam/phishing dataset - [source](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
- [ ] Combined raw human datasets [combine_raw_datasets.py](combine_raw_datasets.py)
- [ ] Cleaning of raw human dataset [cleaning/filter_gibberish.py](cleaning/filter_gibberish.py)
- [x] Get stats of how many 0s and 1s in combined human dataset [human_dataset_stats.py](human_dataset_stats.py)
- [ ] Train classification models
    - [x] BERT / ROBERTa [train/train_lstm.py](train/train_lstm.py)
    - [ ] Random Forest
    - [x] LSTM [train/train_lstm.py](train/train_lstm.py)
    - [ ] SVM
    - [ ] Find best features for ML models to look for 
- [ ] LLM Get phishing categories
    - [x] Fuzzy deduplicate (fix up)
- [ ] Create the prompt dataset for generation
- [ ] Find refusals within prompt dataset
- [x] Generate responses using prompt dataset
- [ ] Testing / Get bypass rates for models, prompts, models/prompts
