import tkinter as tk
from tkinter import ttk
import pickle
import numpy as np
import spacy
import demoji
import re
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords

# Load the pre-trained SpaCy model (GloVe)
nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner"])

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load the ELMo model from TensorFlow Hub
elmo = hub.load("https://tfhub.dev/google/elmo/2").signatures["default"]

# Define text preprocessing function
def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = demoji.replace_with_desc(text)  # Replace emojis with text
    text = re.sub(r'(:[a-zA-Z\s]+:)', r' \1 ', text)  # Add spaces around the shortcode
    text = re.sub(r'(:[a-zA-Z\s]+:)', lambda match: match.group(0).replace(' ', '_'), text)
    text = text.strip()  # Remove leading/trailing spaces
    return text

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Function to extract GloVe embedding using SpaCy
def get_glove_embedding(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    
    # Create GloVe embeddings for the clean text using the SpaCy model
    doc = nlp(text)
    glove_embeddings = np.mean([token.vector for token in doc], axis=0) if doc.vector is not None else np.zeros(300)
    
    return glove_embeddings

# Function to extract ELMo embedding
def get_elmo_embedding(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    
    # Get ELMo embeddings for the clean text
    elmo_embedding = elmo(tf.constant([text]))["elmo"]
    # Average the token embeddings (axis=1 represents tokens, axis=0 represents the batch)
    elmo_embedding_avg = np.mean(elmo_embedding.numpy(), axis=1) if elmo_embedding.shape[0] > 0 else np.zeros(1024)
    
    return elmo_embedding_avg

# Function to combine GloVe and ELMo embeddings (concatenation or averaging)
def combine_embeddings(glove_embedding, elmo_embedding, method="averaging"):
    # Flatten the ELMo embedding if it's 2D (e.g., (1, 1024)) to 1D (e.g., (1024,))
    elmo_embedding = np.squeeze(elmo_embedding)  # This will convert (1, 1024) to (1024,)
    
    if method == "concatenation":
        # Concatenate GloVe and ELMo embeddings
        return np.concatenate([glove_embedding, elmo_embedding])
    elif method == "averaging":
        # Average GloVe and ELMo embeddings
        padding = np.zeros(elmo_embedding.shape[0] - glove_embedding.shape[0])
        glove_embedding = np.concatenate([glove_embedding, padding])
        combined_embedding = (glove_embedding + elmo_embedding) / 2
        return combined_embedding
    else:
        raise ValueError("Method must be 'concatenation' or 'averaging'")

# Load the pre-trained model
def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        return pickle.load(model_file)

# Function to predict sentiment based on the selected model
def predict_sentiment(tweet, model, combo_method="averaging"):
    # Get the embeddings
    glove_embedding = get_glove_embedding(tweet)
    elmo_embedding = get_elmo_embedding(tweet)
    
    # Combine embeddings
    combined_embedding = combine_embeddings(glove_embedding, elmo_embedding, method=combo_method)
    
    # Ensure it's reshaped to (1, n_features) for model prediction
    combined_embedding_reshaped = combined_embedding.reshape(1, -1)
    
    # Make the prediction using the model
    prediction_proba = model.predict_proba(combined_embedding_reshaped)  # Probability for each class
    prediction = model.predict(combined_embedding_reshaped)  # Predicted class label (0 or 1)
    
    return prediction[0], prediction_proba[0]

# Function to update the result when a prediction is made
def on_predict_button_click():
    # Get selected options from the GUI
    language_choice = language_combobox.get()
    combo_choice = combo_combobox.get()
    
    # Set model path based on the selected options
    if language_choice == "English":
        if combo_choice == "Concatenation":
            model_path = "saved_models/English/Glove-Elmo-SVM(concatenate)_English.pkl"
        else:
            model_path = "saved_models/English/Glove-Elmo-SVM(average)_English.pkl"
    elif language_choice == "Tagalog":
        if combo_choice == "Concatenation":
            model_path = "saved_models/Tagalog/Glove-Elmo-SVM(concatenate)_Tagalog.pkl"
        else:
            model_path = "saved_models/Tagalog/Glove-Elmo-SVM(average)_Tagalog.pkl"
    elif language_choice == "Taglish":
        if combo_choice == "Concatenation":
            model_path = "saved_models/Taglish/Glove-Elmo-SVM(concatenate)_Taglish.pkl"
        else:
            model_path = "saved_models/Taglish/Glove-Elmo-SVM(average)_Taglish.pkl"
    elif language_choice == "Mixed":
        if combo_choice == "Concatenation":
            model_path = "saved_models/Mixed/Glove-Elmo-SVM(concatenate)_Mixed.pkl"
        else:
            model_path = "saved_models/Mixed/Glove-Elmo-SVM(average)_Mixed.pkl"
    
    # Load the selected model
    model = load_model(model_path)
    
    # Get the tweet from the user
    tweet = tweet_entry.get()
    
    # Clear previous results
    result_label.config(text="")  # Clear result label before showing new prediction
    
    # Predict sentiment
    predicted_label, predicted_proba = predict_sentiment(tweet, model, combo_method="concatenation" if combo_choice == "Concatenation" else "averaging")
    
    # Update the result label with the new prediction
    result_label.config(text=f"Predicted sentiment: {predicted_label}\nPrediction probabilities: {predicted_proba}")

# Create the main window
root = tk.Tk()
root.title("Sentiment Analysis Prediction GloVe-ELMo-SVM")

# Add widgets for language selection
language_label = tk.Label(root, text="Select Language Model:")
language_label.pack()

language_combobox = ttk.Combobox(root, values=["English", "Tagalog", "Taglish", "Mixed"])
language_combobox.set("English")  # Default value
language_combobox.pack()

# Add widgets for embedding combination method selection
combo_label = tk.Label(root, text="Select Combination Method:")
combo_label.pack()

combo_combobox = ttk.Combobox(root, values=["Concatenation", "Averaging"])
combo_combobox.set("Concatenation")  # Default value
combo_combobox.pack()

# Add widget for tweet input
tweet_label = tk.Label(root, text="Enter a tweet to predict sentiment:")
tweet_label.pack()

tweet_entry = tk.Entry(root, width=50)
tweet_entry.pack()

# Add a button to trigger the prediction
predict_button = tk.Button(root, text="Predict Sentiment", command=on_predict_button_click)
predict_button.pack()

# Label to display the result
result_label = tk.Label(root, text="", justify=tk.LEFT)
result_label.pack()

# Start the Tkinter event loop
root.mainloop()
