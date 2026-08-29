import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from statistics import mode
from sklearn.model_selection import train_test_split

try:
    # Load the dataset
    data = pd.read_csv('updated5.csv')

    # Split the dataset into features and target variable
    X = data.drop('Performance', axis=1)  # Features
    y = data['Performance']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the test dataset
    test_data = pd.read_csv('sabkuch.csv')

    # Check if 'Performance' column is present in test dataset
    if 'Performance' in test_data.columns:
        raise ValueError("Test dataset should not contain the 'Performance' column as it's what you're trying to predict.")

    # Initialize the models
    catboost_model = CatBoostClassifier()
    sgb_model = GradientBoostingClassifier()
    svm_model = SVC(probability=True)  # Ensure probability=True for using predict_proba with SVC

    # Train the models
    catboost_model.fit(X_train, y_train, verbose=False)
    sgb_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)

    # Predict probabilities for each model
    catboost_probs = catboost_model.predict_proba(X_test)[:, 1]
    sgb_probs = sgb_model.predict_proba(X_test)[:, 1]
    svm_probs = svm_model.predict_proba(X_test)[:, 1]

    # Combine predictions from all models using majority voting
    combined_predictions = []
    for prob_catboost, prob_sgb, prob_svm in zip(catboost_probs, sgb_probs, svm_probs):
        combined_pred = mode([1 if prob_catboost >= 0.5 else 0,
                              1 if prob_sgb >= 0.5 else 0,
                              1 if prob_svm >= 0.5 else 0])
        combined_predictions.append(combined_pred)

    # Print the predicted performance for the test dataset
    print("Predicted performance for the test dataset using the combined model:")
    print(combined_predictions)

except FileNotFoundError:
    print("One or both of the files 'updated5.csv' and 'sabkuch.csv' not found.")
except ValueError as e:
    print("ValueError:", e)
except Exception as e:
    print("An error occurred:", e)
