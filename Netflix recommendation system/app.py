

import pandas as pd
import numpy as np
import gradio as gr

# Data Handling & Preprocessing
import pandas as pd  # Handling datasets
import numpy as np   # Numerical computations
import re  # Cleaning text data
import string  # Removing punctuation

# Natural Language Processing (NLP)
import nltk  # NLP utilities
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Text Vectorization (Feature Extraction)
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF method
from sklearn.feature_extraction.text import CountVectorizer  # Bag of Words method

# Similarity Metrics
from sklearn.metrics.pairwise import cosine_similarity  # Cosine similarity
from sklearn.metrics.pairwise import linear_kernel  # Faster cosine similarity

# Dimensionality Reduction (Optional)
from sklearn.decomposition import TruncatedSVD  # Latent Semantic Analysis (LSA)
from sklearn.decomposition import PCA  # Principal Component Analysis (PCA)

# Machine Learning Models (Optional - For Hybrid Models)
from sklearn.neighbors import NearestNeighbors  # kNN for similarity
from sklearn.cluster import KMeans  # Clustering-based recommendation

from sklearn.preprocessing import StandardScaler


# Ensure NLTK stopwords are downloaded (only needed once)
nltk.download('stopwords')
nltk.download('punkt')

df = pd.read_csv('netflix_titles.csv')
print(df.head())
print(df.info())

df['title'].head(30)

"""##Basic analysis

"""

# looking at different columns
df['country'].value_counts().nlargest(25)

#check null values
df.isnull().sum()

# ✅ Fix missing values
df['description'] = df['description'].fillna('')
df['duration'] = df['duration'].fillna('0 min')  # Prevents NaN-related errors
df['country'] = df['country'].fillna('Unknown')

#check again
df.isnull().sum()

"""##Preprocessing"""

# Ensure 'duration' is treated as a string
df['duration'] = df['duration'].astype(str)

# Extract numerical values
df['duration_min'] = df['duration'].apply(lambda x: int(x.replace(" min", "")) if "min" in x else np.nan)
df['num_seasons'] = df['duration'].apply(lambda x: int(x.split()[0]) if "Season" in x else np.nan)

# Fill missing values
df['duration_min'].fillna(0, inplace=True)  # TV Shows will have 0 minutes
df['num_seasons'].fillna(0, inplace=True)  # Movies will have 0 seasons

# Create a categorical feature for movie length
df['duration_category'] = pd.cut(df['duration_min'], bins=[0, 60, 120, np.inf], labels=["Short", "Medium", "Long"])

# Create a binary feature to distinguish TV shows from movies
df['is_tv_show'] = df['num_seasons'].apply(lambda x: 1 if x > 0 else 0)

# Drop the original 'duration' column
df.drop(columns=['duration'], inplace=True)

df.head()

# Ensure 'listed_in' is a string and convert genres into lists
df['listed_in'] = df['listed_in'].astype(str)
df['genres_list'] = df['listed_in'].apply(lambda x: x.split(', '))

# Get all unique genres
all_genres = set(genre for sublist in df['genres_list'] for genre in sublist)

# Create multi-hot encoded genre columns
for genre in all_genres:
    df[genre] = df['genres_list'].apply(lambda x: 1 if genre in x else 0)

# Drop intermediate columns
df.drop(columns=['listed_in', 'genres_list'], inplace=True)

df.head()

# Ensure 'country' column is treated as a string and handle NaNs
df['country'] = df['country'].fillna("Unknown")

# Find the top 10 most common countries
top_countries = df['country'].value_counts().index[:10]

# Replace less frequent countries with 'Other'
df['country_grouped'] = df['country'].apply(lambda x: x if x in top_countries else 'Other')

# Convert into multi-hot encoding (One-Hot Encoding)
country_encoded = pd.get_dummies(df['country_grouped'], prefix='country')

# Merge into the main dataframe
df = pd.concat([df, country_encoded], axis=1)

# Drop the original 'country' columns
df.drop(columns=['country', 'country_grouped'], inplace=True)

df.head()

# Ensure columns are strings and fill missing values
df['director'] = df['director'].astype(str).fillna("")
df['cast'] = df['cast'].astype(str).fillna("")

# Initialize TF-IDF vectorizers
tfidf_director = TfidfVectorizer(stop_words='english', lowercase=True)
tfidf_cast = TfidfVectorizer(stop_words='english', lowercase=True)

# Fit and transform
director_matrix = tfidf_director.fit_transform(df['director'])
cast_matrix = tfidf_cast.fit_transform(df['cast'])

# Compute cosine similarity
director_sim = cosine_similarity(director_matrix, director_matrix)
cast_sim = cosine_similarity(cast_matrix, cast_matrix)

# Fill missing values
df['description'] = df['description'].fillna("")

# Create a TF-IDF vectorizer
tfidf_desc = TfidfVectorizer(stop_words='english')

# Fit and transform
desc_matrix = tfidf_desc.fit_transform(df['description'])

# Compute cosine similarity
text_sim = cosine_similarity(desc_matrix, desc_matrix)

# Select numerical features
num_features = df[['duration_min', 'num_seasons', 'is_tv_show']].fillna(0)

# Scale numeric features
scaler = StandardScaler()
num_features_scaled = scaler.fit_transform(num_features)

# Compute similarity
num_sim = cosine_similarity(num_features_scaled, num_features_scaled)

def categorize_rating(rating):
    kids = ['TV-Y', 'TV-Y7', 'G']
    general = ['PG', 'PG-13', 'TV-PG']
    mature = ['R', 'TV-MA', 'NC-17']

    if rating in kids:
        return 'Kids'
    elif rating in general:
        return 'General'
    elif rating in mature:
        return 'Adult'
    return 'Unknown'

df['age_category'] = df['rating'].apply(categorize_rating)

"""##Model Building"""

# Define feature weights
weights = {
    "genre": 1.5,
    "text": 1.2,
    "duration": 0.7,
    "director": 1.2,
    "cast": 1.0,
    "country": 0.8
}

# Compute final similarity matrix
final_similarity = (
    weights["genre"] * text_sim +  # Text similarity
    weights["text"] * text_sim +   # Description similarity
    weights["duration"] * num_sim +  # Duration similarity
    weights["director"] * director_sim +  # Director similarity
    weights["cast"] * cast_sim  # Cast similarity
)

def get_recommendations(title):
    if title not in df['title'].values:
        return [{"title": "Movie not found in dataset", "release_year": ""}]

    idx = df.index[df['title'] == title].tolist()[0]
    sim_scores = list(enumerate(final_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]  # Top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]

    return df[['title', 'release_year']].iloc[movie_indices].to_dict(orient="records")


iface = gr.Interface(
    fn=get_recommendations,
    inputs=gr.Textbox(label="Enter a Movie Title"),
    outputs=gr.JSON(label="Top 10 Recommended Movies"),
    title="🎬 Netflix Movie Recommendation System",
    description="Enter a movie title and get 10 similar movies based on multiple features.",
)

# Launch Gradio for Hugging Face Spaces
iface.launch(server_name="0.0.0.0", server_port=7860)