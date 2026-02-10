# PICARD: Phishing Intelligence Corpus for Artificial Intelligence Research and Defense: A Massive Approach to Informing Artificial Intelligence Phishing Defenses

AKA Building a dataset of AI-generated phishing messages for use in detecting and stopping AI-generated phishing attacks

## TODO:

- [x] Get unfiltered spam/phishing dataset - [source](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
    - Put in `data/raw/`, leave original filenames
- [x] Combine raw human datasets [combine_raw_datasets.py](cleaning/combine_raw_datasets.py)
    - Creates `data/unfiltered_human_dataset.csv`
- [ ] Cleaning of `data/unfiltered_human_dataset.csv` dataset [cleaning/filter_gibberish.py](cleaning/filter_gibberish.py)
    - Results in cleaned (gibberish free) `data/unfiltered_human_dataset.csv`
    - Removes emails with lots of special characters
    - Can be run with `--dry-run` to instead see how many emails would have been filtered out or kept if ran
- [x] Filtering spam emails out of human dataset [cleaning/filter_spam.py](cleaning/filter_spam.py)
    - Uses an LLM to determine which emails in `data/unfiltered_human_dataset.csv` are just spam, and filters those out
    - Results in `data/combined_human_dataset.csv` and `data/phishing_human_dataset.csv`
- [x] Get stats of how many 0s and 1s in combined human dataset [human_dataset_stats.py](human_dataset_stats.py)
    - Reads from `data/combined_human_dataset` and prints the number of 0 and 1 in the dataset
- [ ] Train classification models
    - [x] BERT / ROBERTa [train/train_lstm.py](train/train_lstm.py)
    - [ ] Random Forest
    - [x] LSTM [train/train_lstm.py](train/train_lstm.py)
    - [ ] SVM
    - [ ] Find best features for ML models to look for 
- [x] LLM Get phishing categories - [spull_ideas.pye](pull_ideas.py)
    - Uses an LLM specified in the file to pull ideas from the dataset of human phishing emails, resulting in an unfiltered list of categories
    - Results in `nemotron_pulled_ideas.csv` or whatever specified filename
- [x] Fuzzy deduplicate pulled categories (fix up) [deduplicate.py](deduplicate.py)
    - Takes `nemotron_pulled_ideas.csv` (or specified filename) and deduplicates
    - Results in `deduplicated_categories.csv`
- [ ] Human creates prompts for generation
- [ ] Take prompts and models to create prompt dataset for generation [create_prompt_dataset.py](create_prompt_dataset.py)
    - Takes the prompts, models, and number of generation for each (either through a cli tool or a file)
- [x] Generate LLM responses using prompt dataset [orchestrator.py](orchestrator.py)
    - Results in `data/outputs.csv`
- [x] Remove refusals from LLM Phishing dataset
    - Takes from `data/outputs.csv`, writes to `data/outputs_cleaned.csv`
    - Can be ran with --dry-run to instead only print the number of refused and normal responses for model/prompt
    - Prints which models/prompts had the most refusals
- [ ] Testing / Get bypass rates for models, prompts, models/prompts
- [ ] Measure bypass rates with graphs and charts