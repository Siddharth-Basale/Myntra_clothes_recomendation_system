import streamlit as st
import re
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download NLTK stopwords if not already available
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Initialize tokenizer
tokenizer = TreebankWordTokenizer()

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    tokens = tokenizer.tokenize(text.lower())  # Tokenize using TreebankWordTokenizer
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

# Load data
data = pd.read_csv('data.csv')

# Preprocess the data
data['Cleaned_ProductName'] = data['ProductName'].apply(clean_text)
data['Cleaned_Description'] = data['Description'].apply(clean_text)
data['Combined_Text'] = data['Cleaned_ProductName'] + " " + data['Cleaned_Description']

# Create TF-IDF vectorizer and matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['Combined_Text'])

# Function to recommend products
def recommend_products(query, top_n=5):
    cleaned_query = clean_text(query)
    query_vector = vectorizer.transform([cleaned_query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]

    recommendations = []
    for idx in top_indices:
        product = {
            "ProductName": data.iloc[idx]['ProductName'],
            "PrimaryColor": data.iloc[idx]['PrimaryColor'],
            "Description": data.iloc[idx]['Description'],
            "Price": data.iloc[idx]['Price (INR)'],
            "Link": f"https://www.myntra.com/{data.iloc[idx]['ProductID']}"
        }
        recommendations.append(product)
    return recommendations

# Streamlit UI
st.title("Product Recommendation System")

# User input
user_query = st.text_input("Enter your search query (e.g., 'men red shirt'):", "")

if user_query:
    st.write(f"### Recommendations for: '{user_query}'")
    
    # Get recommendations
    recommendations = recommend_products(user_query)

    # Display recommendations
    for product in recommendations:
        st.subheader(product["ProductName"])
        st.write(f"**Primary Color:** {product['PrimaryColor']}")
        st.write(f"**Description:** {product['Description']}")
        st.write(f"**Price:** ₹{product['Price']}")
        st.write(f"[View Product]({product['Link']})")
        st.write("---")
