import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("loan_approval_dataset.csv")

# Data Checks
print(df.head())
print(df.info())
print(df.describe())
print("Duplicates:", df.duplicated().sum())
print(df.isnull().sum())
df.columns = df.columns.str.strip()
print(df.columns)

# 1. Loan Status Distribution
sns.countplot(data=df, x='loan_status')
plt.title('Loan Status Distribution')
plt.show()

# 2. Education vs. Loan Status
sns.countplot(data=df, x='education', hue='loan_status')
plt.title('Education vs. Loan Status')
plt.show()

# 3. Loan Amount Distribution
sns.histplot(df['loan_amount'], kde=True)
plt.title('Loan Amount Distribution')
plt.show()

# 4. CIBIL Score vs. Loan Status
sns.boxplot(data=df, x='loan_status', y='cibil_score')
plt.title('CIBIL Score vs. Loan Status')
plt.show()

# 5. Income vs. Loan Amount
sns.scatterplot(data=df, x='income_annum', y='loan_amount', hue='loan_status')
plt.title('Income vs. Loan Amount')
plt.show()

# Preprocessing
print("Before pre-processing:")
print(df['education'].unique())
print(df['self_employed'].unique())
print(df['loan_status'].unique())

encoder = LabelEncoder()
df['loan_status'] = encoder.fit_transform(df['loan_status'])
df['education'] = encoder.fit_transform(df['education'])
df['self_employed'] = encoder.fit_transform(df['self_employed'])

print("\nAfter pre-processing:")
print(df['education'].unique())
print(df['self_employed'].unique())
print(df['loan_status'].unique())

df = df.drop(columns=['loan_id'])

# Split dataset into features (X) and target (y)
X = df.drop(columns=['loan_status'])  # Features
y = df['loan_status']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Predictions and evaluation
y_pred_log = log_reg.predict(X_test)

print("Logistic Regression Metrics:")
print(classification_report(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label='Logistic Regression (AUC = {:.2f})'.format(auc(fpr, tpr)))
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, y_train)

# Predictions and evaluation
y_pred_gbc = gbc.predict(X_test)

print("Gradient Boosting Classifier Metrics:")
print(classification_report(y_test, y_pred_gbc))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gbc))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, gbc.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label='Gradient Boosting (AUC = {:.2f})'.format(auc(fpr, tpr)))
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

from sklearn.model_selection import GridSearchCV

# Hyperparameter Tuning for Logistic Regression
log_reg_params = {
    'C': [0.1, 1, 10],  
    'solver': ['liblinear', 'saga']  
}

# Instantiate GridSearchCV for Logistic Regression
log_reg_grid = GridSearchCV(LogisticRegression(random_state=42), log_reg_params, cv=5, n_jobs=-1, scoring='accuracy')
log_reg_grid.fit(X_train, y_train)

# Display best parameters and best score for Logistic Regression
print("\nBest Parameters for Logistic Regression:", log_reg_grid.best_params_)
print("Best Cross-Validation Score for Logistic Regression: {:.4f}".format(log_reg_grid.best_score_))

# Hyperparameter Tuning for Gradient Boosting Classifier (Optimized)
gbc_params = {
    'n_estimators': [100, 150],  
    'learning_rate': [0.05, 0.1],  
    'max_depth': [3, 4],  
    'subsample': [0.8, 1.0],  
    'min_samples_split': [2, 5]  
}

# Instantiate GridSearchCV for Gradient Boosting Classifier
gbc_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gbc_params, cv=5, n_jobs=-1, scoring='accuracy')
gbc_grid.fit(X_train, y_train)

# Display best parameters and best score for Gradient Boosting
print("Best Parameters for Gradient Boosting Classifier:", gbc_grid.best_params_)
print("Best Cross-Validation Score for Gradient Boosting Classifier: {:.4f}".format(gbc_grid.best_score_))

# Logistic Regression - Best Model
log_best = log_reg_grid.best_estimator_
y_pred_log_best = log_best.predict(X_test)

print("Tuned Logistic Regression Metrics:")
print(classification_report(y_test, y_pred_log_best))

# Gradient Boosting - Best Model
gbc_best = gbc_grid.best_estimator_
y_pred_gbc_best = gbc_best.predict(X_test)

print("Tuned Gradient Boosting Metrics:")
print(classification_report(y_test, y_pred_gbc_best))

# Logistic Regression
train_accuracy_log = accuracy_score(y_train, log_best.predict(X_train))
test_accuracy_log = accuracy_score(y_test, y_pred_log_best)
print(f"Logistic Regression - Training Accuracy: {train_accuracy_log:.4f}")
print(f"Logistic Regression - Test Accuracy: {test_accuracy_log:.4f}")

# Gradient Boosting Classifier
train_accuracy_gbc = accuracy_score(y_train, gbc_best.predict(X_train))
test_accuracy_gbc = accuracy_score(y_test, y_pred_gbc_best)
print(f"Gradient Boosting Classifier - Training Accuracy: {train_accuracy_gbc:.4f}")
print(f"Gradient Boosting Classifier - Test Accuracy: {test_accuracy_gbc:.4f}")
