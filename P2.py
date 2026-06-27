# Email Spam Classifier Project


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# -------------------------
# 1. Load dataset
# -------------------------
# For demo, we create a small dataset inside code.
# In real project, replace this with a CSV file containing emails.
data = {
    'text': [
        "Win money now!!!", 
        "Meeting at 10am tomorrow", 
        "Lowest price guaranteed, buy now", 
        "Project deadline extended", 
        "Congratulations, you won a lottery", 
        "Lunch with team at 1pm"
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']  # ham = not spam
}
df = pd.DataFrame(data)

# -------------------------
# 2. Split dataset
# -------------------------
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -------------------------
# 3. Convert text to numbers
# -------------------------
# Bag of Words model using CountVectorizer
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# -------------------------
# 4. Train Naive Bayes model
# -------------------------
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# -------------------------
# 5. Predictions & Evaluation
# -------------------------
y_pred = model.predict(X_test_counts)

print("\n==============================================")
print("MODEL PERFORMANCE REPORT")
print("==============================================")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("==============================================")

# -------------------------
# 6. Example Prediction
# -------------------------
new_email = ["Claim your free prize now!!!"]
new_email_counts = vectorizer.transform(new_email)
prediction = model.predict(new_email_counts)

print("\n==============================================")
print("NEW EMAIL CLASSIFICATION")
print("==============================================")
print("Email:", new_email[0])
print("Predicted Label:", prediction[0])
print("==============================================")
