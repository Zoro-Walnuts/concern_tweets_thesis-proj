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

# Load the ELMo model from TensorFlow Hub (only required for certain models)
elmo = hub.load("https://tfhub.dev/google/elmo/2").signatures["default"]

# Text preprocessing functions
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

# Functions for loading models
def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        return pickle.load(model_file)

def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as scaler_file:
        return pickle.load(scaler_file)

def load_lsa_model(lsa_model_path):
    with open(lsa_model_path, 'rb') as lsa_file:
        return pickle.load(lsa_file)

# Function to get embeddings (GloVe)
def get_glove_embedding(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    
    # Create GloVe embeddings for the clean text using the SpaCy model
    doc = nlp(text)
    glove_embeddings = np.mean([token.vector for token in doc], axis=0) if doc.vector is not None else np.zeros(300)
    
    return glove_embeddings

# Function to get embeddings (ELMo)
def get_elmo_embedding(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    
    # Get ELMo embeddings for the clean text
    elmo_embedding = elmo(tf.constant([text]))["elmo"]
    elmo_embedding_avg = np.mean(elmo_embedding.numpy(), axis=1) if elmo_embedding.shape[0] > 0 else np.zeros(1024)
    
    return elmo_embedding_avg

# Function to combine embeddings (Concatenation or Averaging)
def combine_embeddings(glove_embedding, elmo_embedding, method="averaging"):
    elmo_embedding = np.squeeze(elmo_embedding)
    
    if method == "concatenation":
        return np.concatenate([glove_embedding, elmo_embedding])
    elif method == "averaging":
        padding = np.zeros(elmo_embedding.shape[0] - glove_embedding.shape[0])
        glove_embedding = np.concatenate([glove_embedding, padding])
        combined_embedding = (glove_embedding + elmo_embedding) / 2
        return combined_embedding
    else:
        raise ValueError("Method must be 'concatenation' or 'averaging'")

# LSA Transformation
def apply_lsa(embedding, lsa_model):
    return lsa_model.transform([embedding])

# Function to select model and options
def select_model_and_options():
    print("Select the language model:")
    print("1. English")
    print("2. Tagalog")
    print("3. Taglish")
    print("4. Mixed")
    language_choice = input("Enter the number corresponding to your choice (1-4): ")

    print("\nSelect the embedding combination method:")
    print("1. Concatenation (GloVe + ELMo)")
    print("2. Averaging (GloVe + ELMo)")
    combo_choice = input("Enter the number corresponding to your choice (1-2): ")

    # Determine paths for models based on user input
    if language_choice == "1":
        model_path = f"saved_models/English/Glove-Elmo-SVM({'concatenate' if combo_choice == '1' else 'average'})_English.pkl"
        scaler_path = "saved_models/Scalers/Scaler_English.pkl"
        lsa_model_path = "saved_models/LSA/LSA_English.pkl"
    elif language_choice == "2":
        model_path = f"saved_models/Tagalog/Glove-Elmo-SVM({'concatenate' if combo_choice == '1' else 'average'})_Tagalog.pkl"
        scaler_path = "saved_models/Scalers/Scaler_Tagalog.pkl"
        lsa_model_path = "saved_models/LSA/LSA_Tagalog.pkl"
    elif language_choice == "3":
        model_path = f"saved_models/Taglish/Glove-Elmo-SVM({'concatenate' if combo_choice == '1' else 'average'})_Taglish.pkl"
        scaler_path = "saved_models/Scalers/Scaler_Taglish.pkl"
        lsa_model_path = "saved_models/LSA/LSA_Taglish.pkl"
    elif language_choice == "4":
        model_path = f"saved_models/Mixed/Glove-Elmo-SVM({'concatenate' if combo_choice == '1' else 'average'})_Mixed.pkl"
        scaler_path = "saved_models/Scalers/Scaler_Mixed.pkl"
        lsa_model_path = "saved_models/LSA/LSA_Mixed.pkl"
    else:
        print("Invalid choice. Defaulting to English.")
        model_path = f"saved_models/English/Glove-Elmo-SVM(concatenate)_English.pkl"
        scaler_path = "saved_models/Scalers/Scaler_English.pkl"
        lsa_model_path = "saved_models/LSA/LSA_English.pkl"
    
    return model_path, scaler_path, lsa_model_path, combo_choice

# Function to predict sentiment
def predict_sentiment(tweet, model, scaler, lsa_model, combo_method="averaging"):
    glove_embedding = get_glove_embedding(tweet)
    elmo_embedding = get_elmo_embedding(tweet)
    
    # Combine embeddings
    combined_embedding = combine_embeddings(glove_embedding, elmo_embedding, method=combo_method)
    
    # Apply LSA transformation if required
    combined_embedding_lsa = apply_lsa(combined_embedding, lsa_model)
    
    # Scale the embeddings using the saved scaler
    combined_embedding_scaled = scaler.transform(combined_embedding_lsa)
    
    # Make the prediction and get the probabilities
    prediction = model.predict(combined_embedding_scaled)
    prediction_proba = model.predict_proba(combined_embedding_scaled)
    
    return prediction[0], prediction_proba[0]

# Main function
if __name__ == "__main__":
    # Select model, scaler, and options based on user input
    model_path, scaler_path, lsa_model_path, combo_choice = select_model_and_options()
    
    # Load the selected model, scaler, and LSA model
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    lsa_model = load_lsa_model(lsa_model_path)
    
    # Get user input for the tweet
    tweet = input("Enter a tweet to predict sentiment: ")
    
    # Predict sentiment and probabilities
    prediction, prediction_proba = predict_sentiment(tweet, model, scaler, lsa_model, combo_method="concatenation" if combo_choice == "1" else "averaging")
    
    # Print the prediction and probabilities
    print(f"Predicted sentiment: {prediction}")
    print(f"Prediction probabilities: {prediction_proba}")
