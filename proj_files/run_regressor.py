import pickle
import numpy as np
import spacy
import demoji
import re
from tkinter import Tk, Label, Button, Entry, Text, OptionMenu, StringVar
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords

# Load the pre-trained Spacy model for text processing
nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner"])

# Load stopwords
stop_words = set(stopwords.words('english'))

# Function to get user input for language/model choice
def select_model_and_scaler(language_choice):
    if language_choice == "English":
        model_path = "saved_models/English/logistic_regression_English.pkl"
        scaler_path = "saved_models/Scalers/RegressorScaler_English.pkl"
    elif language_choice == "Tagalog":
        model_path = "saved_models/Tagalog/logistic_regression_Tagalog.pkl"
        scaler_path = "saved_models/Scalers/RegressorScaler_Tagalog.pkl"
    elif language_choice == "Taglish":
        model_path = "saved_models/Taglish/logistic_regression_Taglish.pkl"
        scaler_path = "saved_models/Scalers/RegressorScaler_Taglish.pkl"
    elif language_choice == "Mixed":
        model_path = "saved_models/Mixed/logistic_regression_Mixed.pkl"
        scaler_path = "saved_models/Scalers/RegressorScaler_Mixed.pkl"
    else:
        model_path = "saved_models/English/logistic_regression_English.pkl"
        scaler_path = "saved_models/Scalers/RegressorScaler_English.pkl"
    
    return model_path, scaler_path

# Load the selected model
def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        return pickle.load(model_file)

# Load the selected scaler
def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as scaler_file:
        return pickle.load(scaler_file)

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

# Function to predict the class of a given tweet and its probabilities
def predict_sentiment(tweet, model, scaler):
    # Preprocess the tweet and extract the embeddings
    avg_embedding = get_avg_embedding(tweet)
    
    # Scale the embeddings using the saved scaler
    avg_embedding_scaled = scaler.transform([avg_embedding])

    # Make the prediction and get the probabilities using the model
    prediction = model.predict(avg_embedding_scaled)
    prediction_proba = model.predict_proba(avg_embedding_scaled)
    
    return prediction[0], prediction_proba[0]

# GUI Logic
def on_predict():
    # Get the tweet from the user input
    tweet = tweet_entry.get()

    # Get the selected model language choice
    language_choice = language_var.get()
    
    # Select model and scaler based on the language choice
    model_path, scaler_path = select_model_and_scaler(language_choice)
    
    # Load the model and scaler
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    
    # Predict sentiment and probabilities
    prediction, prediction_proba = predict_sentiment(tweet, model, scaler)
    
    # Display results
    result_text.delete(1.0, "end")
    result_text.insert("end", f"Predicted sentiment: {prediction}\n")
    result_text.insert("end", f"Prediction probabilities: {prediction_proba}\n")

# Set up the GUI window
root = Tk()
root.title("Sentiment Analysis GloVe-Logistic_Regression")

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
