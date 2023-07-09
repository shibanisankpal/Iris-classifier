import streamlit as st
import random
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Generate a synthetic dataset using the random module
random.seed(42)
n_samples = 100
X, y = datasets.make_classification(n_samples=n_samples, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Use the trained model to make predictions
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Create a Streamlit web app
st.title("Synthetic Dataset Classification")
st.write("This app trains a Random Forest classifier on a synthetic dataset and predicts the labels.")

# Display the accuracy of the model
st.write("Accuracy:", accuracy)

# Display the classification report
classification_rep = classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"])
st.write("Classification Report:")
st.write(classification_rep)

