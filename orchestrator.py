# Takes prompt dataset, generates LLM output with Ollama

import csv
import time
import requests
import sys
import os

url = "http://localhost:"+str(os.environ['OLLAMA_PORT'])+"/api/generate"

input_file = sys.argv[1]

file_path = input_file
output_path = 'outputs/outputs.csv'

new_row = False
while True:
    rows_updated = False
    
    # Read the input CSV
    with open(file_path, mode='r') as file:
        reader = list(csv.DictReader(file))


    for row in reader:

        #if new_row:
        #    get the new iteration from the row

        iterations = int(row['iterations'])

        if iterations > 0:
            #print(f"Model: {row['model']}, Prompt: {row['prompt']}")

            model = row['model']
            prompt = row['prompt']
        

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
            output = response_data.get('response', '').replace('\n', '\\n')
  

            # Write to the output CSV, handling multiline outputs
            with open(output_path, mode='a', newline='', encoding='utf-8') as output_file:
                output_writer = csv.DictWriter(output_file, fieldnames=['model', 'prompt', 'output'])
                if output_file.tell() == 0:  # Add header if file is empty
                    output_writer.writeheader()
                output_writer.writerow({
                    'model': row['model'],
                    'prompt': row['prompt'],
                    'output': output
                })


            row['iterations'] = str(iterations-1)
            #print(str(iterations-1))
            rows_updated = True

            # Write the updated rows back to the input CSV
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=['prompt', 'iterations', 'model'])
                writer.writeheader()
                writer.writerows(reader)

            # Break out of the loop to process the next row
            break

    # Exit the loop if no rows need further processing
    if not rows_updated:
        print("Processing complete.")
        break

    # Optional delay
    #time.sleep(1)
