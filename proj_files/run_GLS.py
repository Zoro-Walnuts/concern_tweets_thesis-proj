import pickle
import numpy as np
import spacy
import demoji
import re
from tkinter import Tk, Label, Button, Entry, Text, OptionMenu, StringVar
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords

# Load the pre-trained Spacy model for text processing
nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner"])

# Load stopwords
stop_words = set(stopwords.words('english'))

# Function to get user input for model, scaler, and LSA selection
def select_model_and_scaler(language_choice):
    if language_choice == "English":
        model_path = "saved_models/English/Glove-LSA-SVM_English.pkl"
        scaler_path = "saved_models/Scalers/GLSScaler_English.pkl"
        lsa_model_path = "saved_models/LSA/LSA_English.pkl"
    elif language_choice == "Tagalog":
        model_path = "saved_models/Tagalog/Glove-LSA-SVM_Tagalog.pkl"
        scaler_path = "saved_models/Scalers/GLSScaler_Tagalog.pkl"
        lsa_model_path = "saved_models/LSA/LSA_Tagalog.pkl"
    elif language_choice == "Taglish":
        model_path = "saved_models/Taglish/Glove-LSA-SVM_Taglish.pkl"
        scaler_path = "saved_models/Scalers/GLSScaler_Taglish.pkl"
        lsa_model_path = "saved_models/LSA/LSA_Taglish.pkl"
    elif language_choice == "Mixed":
        model_path = "saved_models/Mixed/Glove-LSA-SVM_Mixed.pkl"
        scaler_path = "saved_models/Scalers/GLSScaler_Mixed.pkl"
        lsa_model_path = "saved_models/LSA/LSA_Mixed.pkl"
    else:
        model_path = "saved_models/English/Glove-LSA-SVM_English.pkl"
        scaler_path = "saved_models/Scalers/GLSScaler_English.pkl"
        lsa_model_path = "saved_models/LSA/LSA_English.pkl"
    
    return model_path, scaler_path, lsa_model_path

# Load the selected model
def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        return pickle.load(model_file)

# Load the selected scaler
def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as scaler_file:
        return pickle.load(scaler_file)

# Load the selected LSA transformation model
def load_lsa_model(lsa_model_path):
    with open(lsa_model_path, 'rb') as lsa_file:
        return pickle.load(lsa_file)

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

# Function to extract embeddings from text
def get_avg_embedding(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    
    # Create embeddings for the clean text using the Spacy model
    doc = nlp(text)
    embeddings = [token.vector for token in doc]
    avg_embedding = np.mean(embeddings, axis=0) if embeddings else np.zeros(300)
    
    return avg_embedding

# Function to apply LSA transformation
def apply_lsa(embedding, lsa):
    embedding_lsa = lsa.transform([embedding])
    return embedding_lsa

# Function to predict the class and probabilities of a given tweet
def predict_sentiment(tweet, model, scaler, lsa):
    # Preprocess the tweet and extract the embeddings
    avg_embedding = get_avg_embedding(tweet)
    
    # Apply LSA transformation
    avg_embedding_lsa = apply_lsa(avg_embedding, lsa)
    
    # Scale the embeddings using the saved scaler
    avg_embedding_scaled = scaler.transform(avg_embedding_lsa)

    # Make the prediction using the model
    prediction = model.predict(avg_embedding_scaled)
    prediction_proba = model.predict_proba(avg_embedding_scaled)
    
    return prediction[0], prediction_proba[0]

# GUI Logic
def on_predict():
    # Get the tweet from the user input
    tweet = tweet_entry.get()

    # Get the selected model language choice
    language_choice = language_var.get()
    
    # Select model, scaler, and LSA based on the language choice
    model_path, scaler_path, lsa_model_path = select_model_and_scaler(language_choice)
    
    # Load the model, scaler, and LSA transformation
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    lsa = load_lsa_model(lsa_model_path)
    
    # Predict sentiment and probabilities
    prediction, prediction_proba = predict_sentiment(tweet, model, scaler, lsa)
    
    # Display results in the result_text widget
    result_text.delete(1.0, "end")
    result_text.insert("end", f"Predicted sentiment: {prediction}\n")
    result_text.insert("end", f"Prediction probabilities: {prediction_proba}\n")

# Set up the GUI window
root = Tk()
root.title("Sentiment Prediction with GloVe-LSA-SVM")

# Add a label for the language model selection
language_label = Label(root, text="Select language model:")
language_label.pack()

# Set up a dropdown (OptionMenu) for selecting the language model
language_var = StringVar()
language_var.set("English")  # Default value

language_options = ["English", "Tagalog", "Taglish", "Mixed"]
language_menu = OptionMenu(root, language_var, *language_options)
language_menu.pack()

# Add an entry field for the tweet
tweet_label = Label(root, text="Enter a tweet:")
tweet_label.pack()

tweet_entry = Entry(root, width=50)
tweet_entry.pack()

# Add a button to predict sentiment
predict_button = Button(root, text="Predict Sentiment", command=on_predict)
predict_button.pack()

# Add a text box to display the result
result_text = Text(root, height=10, width=50)
result_text.pack()

# Run the GUI
root.mainloop()
