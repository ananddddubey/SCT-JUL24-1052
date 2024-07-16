import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv(r'C:\Users\jidub\OneDrive\Documents\internship\skillcraft\bank+marketing\bank-additional\bank-additional\bank-additional-full.csv', sep=';')

# Function to encode categorical variables
def encode_categorical(df):
    le = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = le.fit_transform(df[column].astype(str))
    return df

# Preprocess the data
X = data.drop('y', axis=1)
y = data['y']

# Encode all categorical variables in X
X = encode_categorical(X)

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get feature importances
importances = dt_classifier.feature_importances_
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importances = feature_importances.sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importances.head(10))