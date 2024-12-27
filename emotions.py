import numpy as np
import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

# Step 1: Load Data
data = pd.read_csv('IMDB-Dataset.csv')
data.sentiment.replace('positive', 1, inplace=True)
data.sentiment.replace('negative', 0, inplace=True)

# Step 2: Text Preprocessing Functions
def clean(text):
    return re.sub(r'<.*?>', '', text)

def is_special(text):
    return ''.join([i if i.isalnum() else ' ' for i in text])

def to_lower(text):
    return text.lower()

def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return " ".join([w for w in words if w not in stop_words])

def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in word_tokenize(text)])

# Apply preprocessing functions to reviews
data.review = data.review.apply(lambda x: stem_txt(rem_stopwords(to_lower(is_special(clean(x))))))

# Step 3: Vectorize Text and Split Data
X = data.review
y = data.sentiment
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(X).toarray()

trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=9)

# Step 4: Train Naive Bayes Classifier
bnb = BernoulliNB()
bnb.fit(trainx, trainy)

# Step 5: Evaluate Model
ypb = bnb.predict(testx)
print("Bernoulli Naive Bayes Accuracy:", accuracy_score(testy, ypb))

# Save the model and vectorizer
pickle.dump(bnb, open('sentiment_model.pkl', 'wb'))
pickle.dump(cv, open('vectorizer.pkl', 'wb'))

# Function to preprocess new input and predict sentiment
def predict_sentiment(review):
    review = stem_txt(rem_stopwords(to_lower(is_special(clean(review)))))
    vectorized = cv.transform([review]).toarray()
    prediction = bnb.predict(vectorized)
    return "Positive" if prediction[0] == 1 else "Negative"

# Test the model with a sample review
new_review = "Terrible movie, complete waste of time."
print("Predicted Sentiment for new review:", predict_sentiment(new_review))
