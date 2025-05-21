# Data handling and machine learning libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Import libraries needed for data handling and machine learning
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Sample data: emails and labels (spam or not spam)
data = {
    'email': [
        'Congratulations! You won a free lottery. Claim now!',
        'Hello, are we meeting tomorrow?',
        'Limited time offer, buy now!',
        'Hi Mom, I will call you later.',
        'Free tickets available, click here.',
        'Are you coming to the party?'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Show the first few rows of data
df.head()


from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Convert emails to feature vectors
X = vectorizer.fit_transform(df['email'])

# Labels
y = df['label']

# Check shape of features matrix
print("Features shape:", X.shape)


from sklearn.model_selection import train_test_split

# Split data into training and testing (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# Predictions karte hain
predictions = model.predict(X_test)

# Evaluation metrics dekhte hain
from sklearn.metrics import accuracy_score, classification_report

print("Predictions:", predictions)
print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))
