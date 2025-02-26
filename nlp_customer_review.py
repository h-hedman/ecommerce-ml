# Author: Hayden Hedman
# Project: Summarizing Customer Review Comments Using NLP
# Date: 2025-02-26
# --------------------------------------------------------------------------------------------------------------
# DESCRIPTION: 
## This script processes customer reviews using Natural Language Processing (NLP) techniques.
## It extracts keywords from customer feedback using tokenization and Term Frequency-Inverse Document Frequency (TF-IDF).
## Sentiment analysis is applied to classify reviews as Positive, Neutral, or Negative.
## The results are visualized to help businesses understand customer sentiment trends.
#
# OBJECTIVE: 
## 1. Identify frequently mentioned words and important keywords in customer reviews.
## 2. Classify reviews by sentiment to track customer satisfaction.
## 3. Provide a visual representation of insights.
#
# Hypothesis:
## H0: Customer review text does not contain meaningful patterns in word importance or sentiment.
## H1: Reviews contain structured sentiment patterns and important words that provide insights into customer experience.
# --------------------------------------------------------------------------------------------------------------
# Libraries
import spacy 
import pandas as pd  
import matplotlib.pyplot as plt 
from collections import Counter  
import string 
from textblob import TextBlob 
from sklearn.feature_extraction.text import TfidfVectorizer 
import random 
# --------------------------------------------------------------------------------------------------------------
# Load the pre-trained spaCy language model
nlp_model = spacy.load("en_core_web_sm")  # Small, efficient English NLP model
# --------------------------------------------------------------------------------------------------------------
# 1. Generate Synthetic Customer Reviews
# --------------------------------------------------------------------------------------------------------------
# Define Example Review Phrases
positive_reviews = ["I love this product!", "Fantastic quality and great service.", "Shipping was super fast!", "Highly recommend!", "Great experience."]
negative_reviews = ["Terrible quality, broke in a week.", "Customer service was unhelpful.", "Slow delivery, very disappointed.", "Would not buy again.", "Misleading description."]
neutral_reviews = ["The product is okay.", "It works as expected.", "Average experience.", "No major complaints.", "It's fine."]

# Generate 100 Synthetic Reviews with Random Sentiment
synthetic_reviews = [random.choice(positive_reviews + negative_reviews + neutral_reviews) for _ in range(100)]

# Convert to DataFrame
customer_reviews_df = pd.DataFrame({"review_id": range(1, 101), "review_text": synthetic_reviews})

# Debugging: Check for missing or empty values
if customer_reviews_df["review_text"].isnull().sum() > 0:
    print(f"Warning: {customer_reviews_df['review_text'].isnull().sum()} missing review(s) detected!")
if (customer_reviews_df["review_text"].str.strip() == "").sum() > 0:
    print("Warning: Some reviews contain only whitespace!")
# --------------------------------------------------------------------------------------------------------------
# 2. Text Preprocessing and Sentiment Analysis
# --------------------------------------------------------------------------------------------------------------
def compute_sentiment_polarity(text):
    """
    Computes sentiment polarity using TextBlob.
    - Positive (> 0)
    - Neutral (≈ 0)
    - Negative (< 0)
    
    Parameters:
    text (str): The input review text.

    Returns:
    float: Sentiment polarity score (-1 to 1).
    """
    return TextBlob(text).sentiment.polarity

# Apply Sentiment Function
customer_reviews_df["sentiment_score"] = customer_reviews_df["review_text"].apply(compute_sentiment_polarity)

# Debugging: Check sentiment score range
if not customer_reviews_df["sentiment_score"].between(-1, 1).all():
    print("Error: Some sentiment scores fall outside the expected range (-1 to 1).")

# Classify Sentiment Categories
customer_reviews_df["sentiment_category"] = customer_reviews_df["sentiment_score"].apply(
    lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral"
)

# Compute Sentiment Distribution
sentiment_distribution = customer_reviews_df["sentiment_category"].value_counts()
print("Sentiment Distribution:")
print(sentiment_distribution)
# --------------------------------------------------------------------------------------------------------------
# 3. Compute TF-IDF for Important Words
# --------------------------------------------------------------------------------------------------------------
# Initialize TF-IDF Vectorizer (n, limit = 20)
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=20) 

# Transform Review Text into TF-IDF Matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(customer_reviews_df["review_text"])

# Convert to DataFrame for Readability
tfidf_results_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Compute Average TF-IDF Scores for Each Word
average_tfidf_scores = tfidf_results_df.mean().sort_values(ascending=False)

# Debugging: Ensure TF-IDF words are extracted correctly
if tfidf_results_df.empty:
    print("Warning: TF-IDF matrix is empty! Check your text preprocessing.")

# Display Top TF-IDF Terms
print("\nTop TF-IDF Words Across Reviews:")
print(average_tfidf_scores.head(10))
# --------------------------------------------------------------------------------------------------------------
# 4. Data Visualization - Sentiment Analysis & TF-IDF
# --------------------------------------------------------------------------------------------------------------
# Sentiment Distribution Plot
plt.figure(figsize=(8, 5))
sentiment_distribution.plot(kind="bar", color=["green", "gray", "red"])
plt.xlabel("Sentiment Category")
plt.ylabel("Number of Reviews")
plt.title("Sentiment Distribution in Customer Reviews")
plt.xticks(rotation=0)
plt.show()

# TF-IDF Keyword Importance Plot
plt.figure(figsize=(10, 5))
average_tfidf_scores.head(10).plot(kind="bar", color="skyblue")
plt.xlabel("Words")
plt.ylabel("Average TF-IDF Score")
plt.title("Top 10 Important Words in Reviews (TF-IDF)")
plt.xticks(rotation=45)
plt.show()
# --------------------------------------------------------------------------------------------------------------
