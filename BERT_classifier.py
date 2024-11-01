# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np
import pandas as pd
from datetime import datetime

# Checking GPU availability to ensure optimal use of resources
if tf.config.list_physical_devices('GPU'):
    print("GPU is available. Optimising for performance.")

# Initialising the BERT model and tokenizer from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Setting up constants
DATA_COLUMN = "sentence"
LABEL_COLUMN = "polarity"
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 128  # Sequence length set based on BERT recommendations for short texts

# Defining function to download and load the dataset
def download_and_load_datasets(force_download=False):
    """This function is downloading the IMDB dataset and setting up train/test dataframes."""
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz", 
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
        extract=True
    )
    # Defining paths for the train and test sets
    train_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "test"))
  
    return train_df, test_df

# Defining helper functions for loading the dataset and handling file processing
def load_directory_data(directory):
    """Loading all files in a directory and extracting sentiment labels from filenames."""
    data = {"sentence": [], "sentiment": []}
    for file_path in os.listdir(directory):
        with open(os.path.join(directory, file_path), "r", encoding="utf-8") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(int(re.match(r"\d+_(\d+)\.txt", file_path).group(1)))
    return pd.DataFrame.from_dict(data)

def load_dataset(directory):
    """Combining positive and negative examples from a directory, adding polarity, and shuffling."""
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Loading the dataset
train, test = download_and_load_datasets()

# Taking a sample to keep the training manageable on standard setups
train = train.sample(5000)
test = test.sample(5000)

# Function to preprocess data for BERT compatibility
def convert_to_input_example(data, data_column, label_column):
    """Tokenising data and preparing it for input into BERT with labels."""
    inputs = tokenizer(
        list(data[data_column]),
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="tf"
    )
    labels = tf.constant(list(data[label_column]), dtype=tf.int32)
    return inputs, labels

# Converting train and test datasets into inputs for BERT
train_inputs, train_labels = convert_to_input_example(train, DATA_COLUMN, LABEL_COLUMN)
test_inputs, test_labels = convert_to_input_example(test, DATA_COLUMN, LABEL_COLUMN)

# Defining BERT model setup
def create_model():
    """Setting up the BERT model with an additional classification layer for fine-tuning."""
    model = TFBertForSequenceClassification.from_pretrained(
        'bert-base-uncased', 
        num_labels=2
    )
    return model

# Model initialisation
model = create_model()

# Setting up training parameters and compilation
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])

# Training the model
print("Starting training...")
current_time = datetime.now()
history = model.fit(
    train_inputs, train_labels, 
    validation_data=(test_inputs, test_labels),
    epochs=3,
    batch_size=BATCH_SIZE
)
print("Training completed in: ", datetime.now() - current_time)

# Evaluation on the test set
print("Evaluating model on test data...")
results = model.evaluate(test_inputs, test_labels, batch_size=BATCH_SIZE)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Defining function for predictions on new sentences
def predict_sentiment(sentences):
    """Predicting sentiment for new sentences and returning probabilities and labels."""
    inputs = tokenizer(
        sentences, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="tf"
    )
    outputs = model(inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()
    labels = ["Negative", "Positive"]
    return [(sentence, labels[np.argmax(prob)], prob) for sentence, prob in zip(sentences, probs)]

# Example predictions
sample_sentences = [
    "The movie was fantastic!",
    "I didn’t enjoy the film.",
    "A truly remarkable performance.",
    "It was dull and uninspiring."
]
predictions = predict_sentiment(sample_sentences)
for sentence, sentiment, prob in predictions:
    print(f"Sentence: '{sentence}' - Sentiment: {sentiment}, Probabilities: {prob}")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
