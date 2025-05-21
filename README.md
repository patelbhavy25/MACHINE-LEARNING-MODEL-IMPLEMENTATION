# MACHINE-LEARNING-MODEL-IMPLEMENTATION

COMPANY : CODTECH IT SOLUTION

NAME : patel bhavy

INTERN ID : CT04DL1286

DOMAIN : PYTHON PROGRAMMING

DURATION : 4 WEEKS

MENTOR : NEELA SANTOSH

# task description

# Introduction


In this project, we build a simple machine learning model to classify emails as spam or ham (not spam). This is a common use case in natural language processing (NLP) and machine learning (ML), especially in email filtering systems used by Gmail, Outlook, etc. The main goal is to take a dataset of emails with their labels (spam/ham), convert the text data into numerical form, and then train a classification model that can detect spam messages.


# Step 1: Importing Required Libraries


We start by importing essential Python libraries:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
pandas is used for handling and organizing data using DataFrames.
train_test_split helps in dividing our data into training and testing sets.
CountVectorizer converts the text emails into numerical feature vectors.
MultinomialNB is the Naive Bayes model, best suited for text classification.
accuracy_score and classification_report are used to evaluate the model’s performance.


# 📧 Step 2: Creating the Dataset


data = {
    'email': [...],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}
df = pd.DataFrame(data)
Here, we create a small sample dataset with 6 email messages and their labels. This is a dummy dataset for demonstration, where:

Spam emails include promotional messages, fake lottery wins, etc.
Ham messages are regular emails like personal messages or meeting reminders.


# Step 3: Text Vectorization


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['email'])
Since machine learning models can’t understand raw text, we use CountVectorizer to convert text into numerical format. This technique is called the Bag of Words (BoW) model. It creates a matrix where each row is an email and each column is a word from the dataset. The values represent how many times each word appears in the email.



# Step 4: Splitting the Data


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
We divide our data into:

70% training data to teach the model.
30% testing data to evaluate how well it performs on unseen examples.
Splitting data is important to prevent overfitting, which happens when a model memorizes training data instead of learning patterns.


# Step 5: Model Training


# Important Note: The code is missing the actual training line. We need to add:
model = MultinomialNB()
model.fit(X_train, y_train)
Here, we initialize a Naive Bayes classifier, which is great for text data and assumes that features (words) are independent. It's a probabilistic model that calculates the likelihood of an email being spam or ham.


# Step 6: Making Predictions


predictions = model.predict(X_test)
Once the model is trained, we use it to predict the labels of the test set (emails it hasn’t seen before).



# Step 7: Evaluating the Model


print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))
We evaluate the performance using:

Accuracy: Tells us the percentage of correct predictions.
Classification Report: Gives precision, recall, and F1-score for both spam and ham classes. These metrics are important when dealing with imbalanced data (e.g., if spam is rare).


# outputs

# shell 1 output




# shell 2 output




# shell 3 output




# shell 4 output




# shell 5 output
