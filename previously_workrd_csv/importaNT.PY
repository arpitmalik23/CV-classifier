import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier

# Read the data
data = pd.read_csv('updated5.csv')

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Performance']), 
                                                    data['Performance'], test_size=0.2, random_state=42)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Initialize XGBoost classifier
xgb_classifier = XGBClassifier()

# Define a broader search space for XGBoost hyperparameters (using param_distributions)
param_distributions_xgb = {
    'learning_rate': np.arange(0.01, 0.3, 0.01),
    'max_depth': range(3, 8),
    'subsample': np.arange(0.6, 1, 0.05),
    'colsample_bytree': np.arange(0.6, 1, 0.05)
}

# Use RandomizedSearchCV for efficient hyperparameter tuning for XGBoost
rand_search_xgb = RandomizedSearchCV(estimator=xgb_classifier, param_distributions=param_distributions_xgb, cv=5, n_jobs=-1)
rand_search_xgb.fit(X_train, y_train)

# Get the best XGBoost parameters and estimator
best_params_xgb = rand_search_xgb.best_params_
best_estimator_xgb = rand_search_xgb.best_estimator_

print("Best XGBoost Parameters:", best_params_xgb)

# Initialize SVM classifier
svm_classifier = SVC()

# Fit SVM classifier
svm_classifier.fit(X_train_imputed, y_train)

# Initialize LightGBM classifier
lgbm_classifier = LGBMClassifier()

# Fit LightGBM classifier
lgbm_classifier.fit(X_train_imputed, y_train)

# Initialize CatBoost classifier
catboost_classifier = CatBoostClassifier()

# Define a broader search space for CatBoost hyperparameters
param_distributions_catboost = {
    'learning_rate': np.arange(0.01, 0.3, 0.01),
    'depth': range(4, 8),
    'subsample': np.arange(0.6, 1, 0.05)
}

# Use RandomizedSearchCV for efficient hyperparameter tuning for CatBoost
rand_search_catboost = RandomizedSearchCV(estimator=catboost_classifier, param_distributions=param_distributions_catboost, cv=5, n_jobs=-1)
rand_search_catboost.fit(X_train, y_train)

# Get the best CatBoost parameters and estimator
best_params_catboost = rand_search_catboost.best_params_
best_estimator_catboost = rand_search_catboost.best_estimator_

print("Best CatBoost Parameters:", best_params_catboost)

# Initialize voting classifier
voting_classifier = VotingClassifier(estimators=[
    ('xgb', best_estimator_xgb),
    ('svm', svm_classifier),
    ('lgbm', lgbm_classifier),
    ('catboost', best_estimator_catboost)
], voting='hard')  # 'hard' voting for simple majority

# Fit voting classifier
voting_classifier.fit(X_train_imputed, y_train)

# Predict using the voting classifier
y_pred_ensemble = voting_classifier.predict(X_test_imputed)

# Calculate accuracy
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f"Ensemble Accuracy: {accuracy_ensemble}")

# Now make predictions on the new data using the ensemble
dfg = pd.read_csv('sabkuch.csv')
features = ['exper', 'incent', 'ticket', 'member', 'earning fm', 'depend', 'work', 'Graduate', 'LC']
ll_test = dfg[features]
ll_test_imputed = imputer.transform(ll_test)
Y_pred_ensemble = voting_classifier.predict(ll_test_imputed)
print(Y_pred_ensemble)

