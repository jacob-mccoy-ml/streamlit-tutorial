import pandas as pd
from time import sleep

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRFClassifier

def get_train_data():
    return pd.read_csv('resources/train_data.csv')

def clean_data(data):
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

    le = LabelEncoder()
    data['Sex'] = le.fit_transform(data['Sex'])
    data['Embarked'] = le.fit_transform(data['Embarked'])

    return data

def train_model(data):
    X = data.drop('Survived', axis=1)
    y = data['Survived']

    model = XGBRFClassifier()

    model.fit(X, y)

    return model

INPUT_CONVERSION = {
    'sex': {'male': 0, 'female': 1},
    'embarked': {'Cherbourg': 0, 'Queenstown': 1, 'Southampton': 2}
}