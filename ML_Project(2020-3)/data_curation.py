import pandas as pd

df = pd.read_csv('car_data.csv', encoding='utf-8')
df['Mileage (kmpl)'] = pd.to_numeric(df['Mileage (kmpl)'], errors="coerce")
df['Engine (CC)'] = pd.to_numeric(df['Engine (CC)'], errors="coerce")
df['Power (bhp)'] = pd.to_numeric(df['Power (bhp)'], errors="coerce")
df['Kilometers_Driven'] = pd.to_numeric(df['Kilometers_Driven'], errors="coerce")
df['Kilometers_Driven']= df['Kilometers_Driven'].where(df['Kilometers_Driven'].between(0,1000000))


print(df.columns)
print('===========================================================================================')
print(df.head())
print('===========================================================================================')
print(df.tail())
print('===========================================================================================')
print(df.describe())
print('===========================================================================================')
print(df.info())
print('===========================================================================================')
print(df.isna().sum())

