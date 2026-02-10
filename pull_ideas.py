import pandas as pd
import time, csv, os, requests, json

debug=False #Add a time delay for development purposes
prompts_file_path = 'data/phishing_human_dataset.csv'
output_file_path = 'nemotron_pulled_ideas.csv'
model = "nemotron_classify_phishing"

# We use Pandas just because its easiest to read/write csv files
df = pd.read_csv(prompts_file_path)

file_exists = os.path.isfile(output_file_path)
id = 0

# Creates the database if necessary
if not file_exists:
    with open(output_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        writer.writerow(['idea'])


url = "http://localhost:"+str(os.environ['OLLAMA_PORT'])+"/api/generate"

# Loop until all iterations for all rows reach 0
while True:

    for index, row in df.iterrows():

        #print(f"Data: {row['prompt']}, Model: {row['model']}")

        #prompt = row['prompt']

        #print("PROMPT:    ", prompt)
        data = {
            "model": model,
            "prompt": "Subject: "+str(row['subject'])+"Body: "+str(row['body']),
            "stream": False
        }

        response = requests.post(url, json=data)
        if response.status_code == 200:
            response_data = response.json()
        else:
            print("Error:", response.status_code, response.text)

        """
        ##output = "Output:  "+prompt # This is where we get output from ollama
        output = response_data.get('response', '')
        print(output)
        id+=1
        f=open("responses/"+str(id)+".txt", "w")
        f.write(output)
        f.close()
        """
        output = response_data.get('response', '')

        #output.replace("\n", "/n")
        print(output)

        with open(output_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([output])
        
            
        if debug: time.sleep(1)

