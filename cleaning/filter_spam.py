# Takes prompt dataset, generates LLM output with Ollama

import csv
import time
import requests
import sys
import os

url = "http://localhost:"+str(os.environ['OLLAMA_PORT'])+"/api/generate"

input_file = "data/unfiltered_human_dataset.csv"

combined_output_path = 'data/combined_human_dataset.csv'
phishing_output_path = 'data/phishing_human_dataset.csv'

prompt = "Is this a spam email?"
model = "deepseek-r1:14b"


def error_log(line):
    with open("filter_spam_error.log", "a") as file:
        file.write(line)

line = 0
new_row = False
while True:
    rows_updated = False
    
    # Read the input CSV
    with open(input_file, mode='r') as file:
        reader = list(csv.DictReader(file))

    for row in reader:
            line+=1
            new_label = 0
        
            # Labeled as phishing/spam in the datasets
            if label == 1:
                
                text = text + row['body']
                label = row['label']

                # Prepare the request payload
                data = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }

                # Send the request to the API
                response = requests.post(url, json=data)
                if response.status_code == 200:
                    response_data = response.json()
                else:
                    print("Error:", response.status_code, response.text)
                    continue
                
                # Get the multiline output and format new lines as \n
                output = response_data.get('response', '')

                if len(output.split()) > 1:
                    error_log(line)
                    break

                # We store emeials which are labeled as 1 and also are marked as not spam by the LLM
                if output != "Spam":
                    new_label = 1

                    with open(phishing_output_path, mode='a', newline='', encoding='utf-8') as output_file:
                        output_writer = csv.DictWriter(output_file, fieldnames=['model', 'prompt', 'output'])
                        if output_file.tell() == 0:  # Add header if file is empty
                            output_writer.writeheader()
                        output_writer.writerow({
                            'body': row['body'],
                            'label': new_label
                        })

            with open(combined_output_path, mode='a', newline='', encoding='utf-8') as output_file:
                output_writer = csv.DictWriter(output_file, fieldnames=['model', 'prompt', 'output'])
                if output_file.tell() == 0:  # Add header if file is empty
                    output_writer.writeheader()
                output_writer.writerow({
                    'body': row['body'],
                    'label': new_label
                })

    # Exit the loop if no rows need further processing
    if not rows_updated:
        print("Processing complete.")
        break

    # Optional delay
    #time.sleep(1)
