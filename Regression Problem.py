# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve

# Load dataset
df = pd.read_csv('laptop_prices.csv')

# Data Checks
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# 1. Distribution of the Target Variable (Price)
plt.figure(figsize=(8, 6))
sns.histplot(df['Price_euros'], kde=True, color='blue', bins=30)
plt.title('Price Distribution')
plt.xlabel('Price (Euros)')
plt.ylabel('Frequency')
plt.show()

# 2. RAM vs. Price
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Ram', y='Price_euros', palette='coolwarm')
plt.title('Price Distribution by RAM')
plt.xlabel('RAM (GB)')
plt.ylabel('Price (Euros)')
plt.show()

# 3. Average Price by Company
avg_price_by_company = df.groupby('Company')['Price_euros'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=avg_price_by_company.index, y=avg_price_by_company.values, palette='viridis')
plt.title('Average Price by Company')
plt.xlabel('Company')
plt.ylabel('Average Price (Euros)')
plt.xticks(rotation=45)
plt.show()

# 4. Weight vs. Price (Scatter Plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Weight', y='Price_euros', hue='TypeName', palette='deep', alpha=0.8)
plt.title('Weight vs. Price by Laptop Type')
plt.xlabel('Weight (kg)')
plt.ylabel('Price (Euros)')
plt.legend(title='Laptop Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# 5. Countplot of Laptops by Type
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='TypeName', palette='muted', order=df['TypeName'].value_counts().index)
plt.title('Laptop Count by Type')
plt.xlabel('Laptop Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Preprocessing
# Encoding categorical variables
label_encoders = {}
categorical_columns = df.select_dtypes(include='object').columns

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature Engineering
df['Screen_Resolution'] = df['ScreenW'] * df['ScreenH']  
df['Storage_Total'] = df['PrimaryStorage'] + df['SecondaryStorage']  

# Dropping unnecessary columns
df.drop(['Product', 'ScreenW', 'ScreenH', 'PrimaryStorage', 'SecondaryStorage'], axis=1, inplace=True)

# Feature Scaling
scaler = StandardScaler()
scaled_features = ['Inches', 'Ram', 'Weight', 'CPU_freq', 'Screen_Resolution', 'Storage_Total']

df[scaled_features] = scaler.fit_transform(df[scaled_features])

# Splitting the dataset
X = df.drop('Price_euros', axis=1)
y = df['Price_euros']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluation
print("\nLinear Regression Metrics:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_lr):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred_lr):.4f}")

# Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation
print("\nRandom Forest Metrics (Before Tuning):")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_rf):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred_rf):.4f}")

# Hyperparameter Tuning with GridSearchCV
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

rf_grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=rf_param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

rf_grid_search.fit(X_train, y_train)

# Best parameters and cross-validation score
print("\nBest Parameters for Random Forest:")
print(rf_grid_search.best_params_)
print(f"Best Cross-Validation Score: {rf_grid_search.best_score_:.4f}")


# Re-train Random Forest with Best Parameters
rf_best = rf_grid_search.best_estimator_
y_pred_rf_tuned = rf_best.predict(X_test)

# Evaluation After Tuning
print("\nRandom Forest Metrics (After Tuning):")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_rf_tuned):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred_rf_tuned):.4f}")

# Comparison of Metrics Before and After Tuning
print("\nModel Comparison:")
print(f"Linear Regression R2 Score: {r2_score(y_test, y_pred_lr):.4f}")
print(f"Random Forest R2 Score (Before Tuning): {r2_score(y_test, y_pred_rf):.4f}")
print(f"Random Forest R2 Score (After Tuning): {r2_score(y_test, y_pred_rf_tuned):.4f}")

# Plot Predictions vs Actual Prices
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_lr, label='Linear Regression', alpha=0.6)
plt.scatter(y_test, y_pred_rf_tuned, label='Random Forest (Tuned)', alpha=0.6, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.show()

# Feature Importance
importances = rf_best.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title('Feature Importance - Random Forest Regressor')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()


# Learning Curves for Random Forest Regressor
train_sizes, train_scores, test_scores = learning_curve(
    rf_best, X_train, y_train, cv=3, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 5), scoring='r2'
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, label='Training Score', color='blue')
plt.plot(train_sizes, test_scores_mean, label='Cross-Validation Score', color='red')
plt.title('Learning Curves - Random Forest Regressor')
plt.xlabel('Training Set Size')
plt.ylabel('R2 Score')
plt.legend(loc='best')
plt.show()


# Linear Regression Metrics
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_r2 = r2_score(y_test, y_pred_lr)
lr_mae = mean_absolute_error(y_test, y_pred_lr)

print("\nLinear Regression Performance Metrics:")
print(f"RMSE: {lr_rmse:.4f}")
print(f"R2 Score: {lr_r2:.4f}")
print(f"MAE: {lr_mae:.4f}")

# Random Forest Metrics
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_tuned))
rf_r2 = r2_score(y_test, y_pred_rf_tuned)
rf_mae = mean_absolute_error(y_test, y_pred_rf_tuned)

print("\nRandom Forest Regressor Performance Metrics (After Tuning):")
print(f"RMSE: {rf_rmse:.4f}")
print(f"R2 Score: {rf_r2:.4f}")
print(f"MAE: {rf_mae:.4f}")
