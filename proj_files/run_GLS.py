import pickle
import numpy as np
import spacy
import demoji
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords

# Load the pre-trained Spacy model
nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner"])

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load the pre-trained model and scaler
model_path = "saved_models/English/Glove-LSA-SVM_English.pkl"
scaler_path = "saved_models/Scalers/GLSScaler_English.pkl"
lsa_model_path = "saved_models/LSA/LSA_English.pkl"

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open(lsa_model_path, 'rb') as lsa_file:
    lsa = pickle.load(lsa_file)

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
def apply_lsa(embedding):
    embedding_lsa = lsa.transform([embedding])
    return embedding_lsa

# Function to predict the class of a given tweet
def predict_sentiment(tweet):
    # Preprocess the tweet and extract the embeddings
    avg_embedding = get_avg_embedding(tweet)
    
    # Apply LSA transformation
    avg_embedding_lsa = apply_lsa(avg_embedding)
    
    # Scale the embeddings using the saved scaler
    avg_embedding_scaled = scaler.transform(avg_embedding_lsa)

    # Make the prediction using the model
    prediction = model.predict_proba(avg_embedding_scaled)
    
    return prediction[0]

# Main function to interact with the user
if __name__ == "__main__":
    tweet = input("Enter a tweet to predict sentiment: ")
    prediction = predict_sentiment(tweet)
    
    # Print the prediction
    print(f"Predicted sentiment: {prediction}")
