# ARTI308-Lab2-HeartDisease
Problem Type: 
This is a supervised classification problem.

Target Variable:
The target variable is heart disease presence, where:
1 = Patient has heart disease
0 = Patient does not have heart disease

Problem Description:
The objective of this project is to build a machine learning classification model that predicts whether a patient has heart disease based on several clinical and physiological features such as age, sex, chest pain type, resting blood pressure, cholesterol level, fasting blood sugar and maximum heart rate.
The model aims to assist in early diagnosis and support medical decision-making by identifying high-risk patients using data-driven techniques.

import pandas as pd
from google.colab import files

uploaded = files.upload()

filename = next(iter(uploaded))
df = pd.read_csv(filename)

print("Shape of the dataset:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nColumn names and data types:\n", df.dtypes)
