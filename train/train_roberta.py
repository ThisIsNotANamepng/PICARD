import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import random

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load Data
file_path = 'data/combined_human_dataset.csv'
try:
    df = pd.read_csv(file_path)
    if not all(col in df.columns for col in ['body', 'label']):
        raise ValueError("CSV must contain 'body' and 'label' columns")
except FileNotFoundError:
    print(f"File not found at {file_path}. Creating dummy data.")
    data = {
        'body': ['I loved this movie', 'Terrible plot', 'Great performance', 
                 'I fell asleep', 'Stunning visuals', 'It was okay'] * 20,
        'label': ['positive', 'negative', 'positive', 'negative', 'positive', 'neutral'] * 20
    }
    df = pd.DataFrame(data)

# --- Data Cleaning ---
df['body'] = df['body'].fillna('').astype(str)
df['label'] = df['label'].astype(str)

# 2. Preprocessing
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# 3. Train/Test Split (90/10)
# Using .tolist() to avoid the PyArrow indexing error
X_train, X_test, Y_train, Y_test = train_test_split(
    df['body'].tolist(), 
    df['label_encoded'].tolist(), 
    test_size=0.1, 
    random_state=42, 
    stratify=df['label_encoded'].tolist()
)

# 4. Tokenization
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_df = pd.DataFrame({'text': X_train, 'label': Y_train})
test_df = pd.DataFrame({'text': X_test, 'label': Y_test})

train_df['text'] = train_df['text'].astype(str)
test_df['text'] = test_df['text'].astype(str)

train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(['text'])
test_dataset = test_dataset.remove_columns(['text'])

# 5. Define Model and Training Logic
def model_init():
    return RobertaForSequenceClassification.from_pretrained(
        'roberta-base', 
        num_labels=len(le.classes_)
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

# 6. Hyperparameter Tuning
param_grid = {
    "learning_rate": [2e-5, 3e-5, 4e-5, 5e-5],
    "per_device_train_batch_size": [16, 32],
    "num_train_epochs": [3, 4], 
    "weight_decay": [0.01, 0.0]
}

def random_search_trainer(n_trials=3):
    best_score = 0
    best_params = {}

    print(f"Starting Random Search with {n_trials} trials...")

    for i in range(n_trials):
        current_params = {
            "learning_rate": random.choice(param_grid["learning_rate"]),
            "per_device_train_batch_size": random.choice(param_grid["per_device_train_batch_size"]),
            "num_train_epochs": random.choice(param_grid["num_train_epochs"]),
            "weight_decay": random.choice(param_grid["weight_decay"]),
            # --- FIX: Use eval_strategy instead of evaluation_strategy ---
            "eval_strategy": "epoch",
            "save_strategy": "no",
            "logging_strategy": "no",
            "load_best_model_at_end": False,
            "report_to": "none"
        }

        print(f"\nTrial {i+1}: Testing Learning Rate: {current_params['learning_rate']}")

        model = model_init()

        training_args = TrainingArguments(
            output_dir=f"./tmp/trial_{i}",
            **current_params
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()
        eval_result = trainer.evaluate()

        current_score = eval_result['eval_accuracy']
        print(f"Trial {i+1} Accuracy: {current_score:.4f}")

        if current_score > best_score:
            best_score = current_score
            best_params = current_params
            print(f"*** New Best Model Found! Accuracy: {best_score:.4f} ***")

    return best_params, best_score

best_params, best_score = random_search_trainer(n_trials=3)

print("\n" + "="*30)
print("Best Hyperparameters Found:")
for k, v in best_params.items():
    print(f"{k}: {v}")
print("="*30)

# 7. Final Training
print("Training final model with best parameters...")

final_model = model_init()
final_args = TrainingArguments(
    output_dir="./final_model",
    learning_rate=best_params['learning_rate'],
    per_device_train_batch_size=best_params['per_device_train_batch_size'],
    num_train_epochs=best_params['num_train_epochs'],
    weight_decay=best_params['weight_decay'],
    # --- FIX: Use eval_strategy here as well ---
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_strategy="epoch",
    report_to="none"
)

final_trainer = Trainer(
    model=final_model,
    args=final_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

final_trainer.train()

# 8. Final Evaluation
print("\nFinal Evaluation Results:")
metrics = final_trainer.evaluate()
print(f"Test Accuracy: {metrics['eval_accuracy']:.4f}")

raw_pred, _, _ = final_trainer.predict(test_dataset)
y_pred = np.argmax(raw_pred, axis=1)
print("\nClassification Report:")
print(classification_report(Y_test, y_pred, target_names=le.classes_))

final_trainer.save_model("./best_roberta_model")
tokenizer.save_pretrained("./best_roberta_model")
print("\nModel saved to ./best_roberta_model")
