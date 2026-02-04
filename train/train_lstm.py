import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras_tuner as kt

# 1. Load Data
file_path = 'data/combined_human_dataset.csv'

try:
    df = pd.read_csv(file_path)
    if not all(col in df.columns for col in ['body', 'label']):
        raise ValueError("CSV must contain 'body' and 'label' columns")
except FileNotFoundError:
    print(f"File not found at {file_path}. Creating dummy data.")
    data = {
        'body': [
            'I loved this movie', 'Terrible plot', 'Great performance', 
            'I fell asleep', 'Stunning visuals', np.nan, 'Just okay'
        ] * 20, # Added a NaN to simulate the error
        'label': ['positive', 'negative', 'positive', 'negative', 'positive', 'neutral', 'neutral'] * 20
    }
    df = pd.DataFrame(data)

# --- CRITICAL FIX: Data Cleaning ---
# Replace NaN with empty strings and ensure everything is a string type
df['body'] = df['body'].fillna('').astype(str)
# Also ensure labels are strings if they might be mixed types
df['label'] = df['label'].astype(str) 
# ---------------------------------

# 2. Preprocessing
# Encode labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Parameters for Tokenization
MAX_NB_WORDS = 5000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token="<OOV>")
# This line will now work without AttributeError
tokenizer.fit_on_texts(df['body'].values) 

X = tokenizer.texts_to_sequences(df['body'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

Y = df['label_encoded'].values

# 3. Train/Test Split (90/10)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, random_state=42, stratify=Y
)

# 4. Hyperparameter Tuning
def model_builder(hp):
    model = Sequential()
    hp_units = hp.Int('units', min_value=64, max_value=256, step=64)
    hp_embedding_dim = hp.Int('embedding_dim', min_value=64, max_value=256, step=64)

    model.add(Embedding(MAX_NB_WORDS, hp_embedding_dim))

    # Tuning Dropout for regularization (prevents overfitting)
    hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)
    model.add(SpatialDropout1D(hp_dropout))

    # Tuning Bidirectional layer (standard practice in text classification research)
    hp_bidirectional = hp.Boolean('bidirectional')

    if hp_bidirectional:
        model.add(Bidirectional(LSTM(hp_units)))
    else:
        model.add(LSTM(hp_units))

    model.add(Dropout(hp_dropout))
    model.add(Dense(len(le.classes_), activation='softmax'))

    # Tuning Learning Rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.RandomSearch(
    model_builder,
    objective='val_accuracy',
    max_trials=5,
    directory='tuning_dir',
    project_name='lstm_fixed'
)

stop_early = EarlyStopping(monitor='val_loss', patience=3)

# Create a validation split from training data for tuning
X_tune_train, X_val, Y_tune_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.1, random_state=42, stratify=Y_train
)

print("Starting Hyperparameter Search...")
tuner.search(
    X_tune_train, Y_tune_train, 
    epochs=5, 
    validation_data=(X_val, Y_val), 
    callbacks=[stop_early]
)

# Get best hyperparameters and build final model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

print("Training final model...")
history = model.fit(
    X_train, Y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    callbacks=[stop_early],
    verbose=1
)

# Evaluation
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test Accuracy: {accuracy}')
