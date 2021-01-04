import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# read preprocessed dataset
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('preprocessing_data.csv')

# MinMaxScaler

np.random.seed(1)
data_df = pd.DataFrame({
    'Year': df['Year'],
    'Trans': df['Transmission'],
    'Owner_Type': df['Owner_Type'],
    'Engine': df['Engine (CC)'],
    'Power': df['Power (bhp)'],
})

scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(data_df)
scaled_df = pd.DataFrame(scaled_df, columns=['Year', 'Trans', 'Owner_Type', 'Engine', 'Power'])

# MultipleRegression
x = df[['Year', 'Transmission', 'Owner_Type', 'Engine (CC)', 'Power (bhp)']]
y = df[['Price']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
reg = LinearRegression()
reg.fit(x_train, y_train)

y_predict = reg.predict(x_test)
plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple Regression")
print('Intercept: \n', reg.intercept_)  # weight, coefficient
print('Coefficients: \n', reg.coef_)  # offset,intercept
print("\nThe Car Price Prediction Score :", reg.score(x_train, y_train))
plt.show()

# k-fold evaluation
kfold = KFold(n_splits=5, shuffle=True)  # k=5
results = cross_val_score(reg, x, y, cv=kfold)
print('\nValidation Score :', results)
print('\nThe mean of Validation score :', sum(results) / 5)

# TKinter GUI

root = tk.Tk()

canvas1 = tk.Canvas(root, width=520, height=300)
canvas1.pack()

Intercept_result = ('Intercept : ', reg.intercept_)
label_Intercept = tk.Label(root, text=Intercept_result, justify='center')
canvas1.create_window(260, 190, window=label_Intercept)

Coefficients_result = ('Coefficients : ', reg.coef_)
label_Coefficients = tk.Label(root, text=Coefficients_result, justify='center')
canvas1.create_window(260, 220, window=label_Coefficients)

label1 = tk.Label(root, text='Type year : ')
canvas1.create_window(100, 30, window=label1)

entry1 = tk.Entry(root)  # create 1st entry box
canvas1.create_window(270, 30, window=entry1)

label2 = tk.Label(root, text=' Type trans : ')
canvas1.create_window(100, 50, window=label2)

entry2 = tk.Entry(root)  # create 2nd entry box
canvas1.create_window(270, 50, window=entry2)

label3 = tk.Label(root, text=' Type owner_type : ')
canvas1.create_window(100, 70, window=label3)

entry3 = tk.Entry(root)  # create 3rd entry box
canvas1.create_window(270, 70, window=entry3)

label4 = tk.Label(root, text=' Type engine : ')
canvas1.create_window(100, 90, window=label4)

entry4 = tk.Entry(root)  # create 4th entry box
canvas1.create_window(270, 90, window=entry4)

label5 = tk.Label(root, text=' Type power : ')
canvas1.create_window(100, 110, window=label5)

entry5 = tk.Entry(root)  # create 5th entry box
canvas1.create_window(270, 110, window=entry5)


def values():
    global New_year  # our 1st input variable
    New_year = float(entry1.get())

    global New_trans  # our 2nd input variable
    New_trans = float(entry2.get())

    global New_owner  # our 3rd input variable
    New_owner = float(entry3.get())

    global New_engine  # our 4th input variable
    New_engine = float(entry4.get())

    global New_power  # our 5th input variable
    New_power = float(entry5.get())

    Prediction_result = ('Predicted Price : ', reg.predict([[New_year, New_trans, New_owner, New_engine, New_power]]))
    label_Prediction = tk.Label(root, text=Prediction_result)
    canvas1.create_window(260, 260, window=label_Prediction)


button1 = tk.Button(root, text='Predict Price', command=values,
                    bg='green')  # button to call the 'values' command above
canvas1.create_window(270, 150, window=button1)
