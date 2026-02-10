# Takes the raw csv files fomr the human datasets and removes all uneeded column

import pandas as pd

def clean_csv(input_file, output_file, just_positive=False):
    #just_positive=False #True if you want to grab the messages that are only phishing

    # Read the CSV file
    df = pd.read_csv(input_file, dtype=str)
    
    # This filters the resulting combined dataset 
    if just_positive: df = df[df['label'] == '1']
    
    # Rename columns
    df = df.rename(columns={'subject': 'subject', 'author': 'sender', 'content': 'body'})
    
    # Ensure required columns exist, filling missing ones with empty strings
    #required_columns = ['subject', 'sender', 'body']
    required_columns = ['body', 'label']

    for col in required_columns:
        if col not in df:
            df[col] = ''
    
    # Keep only the required columns
    df = df[required_columns]
    
    # Replace line breaks with \n
    df = df.map(lambda x: x.replace('\n', ' ').replace('\r', ' ') if pd.notnull(x) else x)
    
    # Append to the output CSV file
    df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)


combined_output_csv = 'data/unfiltered_combined_human_dataset.csv'

clean_csv("data/raw/SpamAssasin.csv", combined_output_csv)
clean_csv("data/raw/CEAS_08.csv", combined_output_csv)
clean_csv("data/raw/Ling.csv", combined_output_csv)
clean_csv("data/raw/Nazario.csv", combined_output_csv)
clean_csv("data/raw/Nigerian_Fraud.csv", combined_output_csv)
