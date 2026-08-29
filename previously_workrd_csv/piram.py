import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('updated5.csv')

# Split the dataset into features and target variable
X = data.drop('Performance', axis=1) # Features
y = data['Performance'] # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Classifier
sgb_model = GradientBoostingClassifier()

# Train the model
sgb_model.fit(X_train, y_train)

# Predict the performance on the test set
y_pred = sgb_model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Load the test dataset
test_data = pd.read_csv('sabkuch.csv')

# Make predictions on the test dataset
predictions = sgb_model.predict(test_data)

# Print the predicted performance
print("Predicted performance for the test dataset:")
print(predictions)
