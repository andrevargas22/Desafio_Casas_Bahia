import numpy as np 
import pandas as pd

# importing dataset
df = pd.read_csv('data/1. raw/train.csv')

# fill missing Age values with the mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Drop the Cabin column
df.drop('Cabin', axis=1, inplace=True)

# fill missing Embarked values with the mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df.replace({'Sex': {'male':0, 'female':1}}, inplace=True)
df.replace({'Embarked': {'S':0, 'C':1, 'Q':2}}, inplace=True)

def detect_outliers(df, features):
    outliers = []
    for feature in features:
        if df[feature].dtype.kind in 'bifc':  # Check if feature is numeric
            Q1 = np.percentile(df[feature], 25)
            Q3 = np.percentile(df[feature], 75)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            outliers.extend(df[(df[feature] < Q1 - outlier_step) | (df[feature] > Q3 + outlier_step)].index)
    return outliers

# Detect outliers
features = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']  # Remove non-numeric features
outliers = detect_outliers(df, features)

df = df.drop(outliers).reset_index(drop=True)

df.select_dtypes(exclude='object')

df = df[["Pclass", "Sex", "SibSp", "Parch", "Fare"]]

print(df)


