import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'tested.csv'  # Update this to your dataset path
titanic_data = pd.read_csv(file_path)

# Data Cleaning and Preprocessing
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Fare'].fillna(titanic_data['Fare'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data['Cabin_Indicator'] = titanic_data['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)

# Define features and target
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin_Indicator']
X = titanic_data[features]
y = titanic_data['Survived']

# Preprocessing for numerical and categorical features
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Embarked']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])

# Create a simpler model to address the perfect accuracy issue
model = LogisticRegression(max_iter=1000, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])

# Cross-validation to evaluate model
cv_scores = cross_val_score(clf, X, y, cv=5)
print(f'CV Accuracy Scores: {cv_scores}')
print(f'CV Accuracy Mean: {np.mean(cv_scores)}')

# Data Visualization
# Switching from 'Sex' to 'Embarked' for categorical visualization
plt.figure(figsize=(6,4))
sns.countplot(x='Embarked', hue='Survived', data=titanic_data)
plt.title('Survival Rate by Embarkation Point')
plt.xlabel('Embarkation Point')
plt.ylabel('Count')
plt.show()

# Additional visualizations
# Passengers by class
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', data=titanic_data)
plt.title('Passenger Count by Class')
plt.show()

# Age vs Fare Paid, colored by Survival
plt.figure(figsize=(8,6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=titanic_data)
plt.title('Age vs Fare Paid, Colored by Survival')
plt.show()