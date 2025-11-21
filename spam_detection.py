import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load and prepare data
df = pd.read_csv('spam_ham_dataset.csv', encoding='utf-8')
X = df['text']
y = df['label_num']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# User-defined message checking loop
print("Spam Detector Ready! Type your message (or type 'exit' to quit):")
while True:
    user_message = input("Enter your message: ")
    if user_message.lower() == 'exit':
        print("Exiting Spam Detector.")
        break
    msg_vec = vectorizer.transform([user_message])
    prediction = model.predict(msg_vec)[0]
    result = "Spam" if prediction == 1 else "Ham"
    print(f"Result: {result}\n")