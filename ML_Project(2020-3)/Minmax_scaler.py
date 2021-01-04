import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# read dataset
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('preprocessing_data.csv')

# MinMaxScaler
np.random.seed(1)
data_df = pd.DataFrame({
    'Year': df['Year'],
    'Owner_Type': df['Owner_Type'],
    'Engine': df['Engine (CC)'],
    'Power': df['Power (bhp)'],
})

scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(data_df)
scaled_df = pd.DataFrame(scaled_df, columns=['Year', 'Owner_Type', 'Engine', 'Power'])
# print(scaled_df)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
ax1.set_title('Befor Scaling')
sns.kdeplot(data_df['Year'], ax=ax1)
ax2.set_title('After MinMax Scaler')
sns.kdeplot(scaled_df['Year'], ax=ax2)
plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
ax1.set_title('Befor Scaling')
sns.kdeplot(data_df['Owner_Type'], ax=ax1)
ax2.set_title('After MinMax Scaler')
sns.kdeplot(scaled_df['Owner_Type'], ax=ax2)
plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
ax1.set_title('Befor Scaling')
sns.kdeplot(data_df['Engine'], ax=ax1)
ax2.set_title('After MinMax Scaler')
sns.kdeplot(scaled_df['Engine'], ax=ax2)
plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
ax1.set_title('Befor Scaling')
sns.kdeplot(data_df['Power'], ax=ax1)
ax2.set_title('After MinMax Scaler')
sns.kdeplot(scaled_df['Power'], ax=ax2)
plt.show()
