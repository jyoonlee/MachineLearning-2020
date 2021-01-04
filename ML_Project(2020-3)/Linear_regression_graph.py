import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# read dataset
data = pd.read_csv('train-data.csv', encoding='utf-8')

# copy
df = data.copy()

# drop unnecessary data
df = df.drop(['New_Price'], 1)

# Preprocessing
# change value 0 to NaN
df.replace('null ', 0, inplace=True)
df.replace('?', 0, inplace=True)
reg=LinearRegression()

# Detect Outlier
df['Kilometers_Driven']=df['Kilometers_Driven'].astype(float)
df['Kilometers_Driven'] = df['Kilometers_Driven'].where(df['Kilometers_Driven'].between(0, 1000000))
df = df.fillna(0)

# name, location
df2=df.drop(['Year', 'Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Mileage (kmpl)','Engine (CC)','Power (bhp)','Seats','Price'] ,1)


# year linear regression
df3=df.drop(['Name','Location', 'Kilometers_Driven','Fuel_Type','Transmission',
             'Owner_Type','Mileage (kmpl)','Engine (CC)','Power (bhp)','Seats'] ,1)
mask_zero=(df3['Year']==0)
mask_non=(df3['Year']!=0)
df_zero=df3[mask_zero]
df_non=df3[mask_non]
x_t=df_non['Price']
y_d=df_non['Year']
x_t1=np.atleast_2d(x_t)
x_t2=np.transpose(x_t1)
y_d1=np.atleast_2d(y_d)
y_d2=np.transpose(y_d1)
y_d2=y_d2.astype(float)
reg.fit(x_t2, y_d2)
px=np.array([x_t2.min()-1,x_t2.max()+1])
py=reg.predict(px[:,np.newaxis])
plt.scatter(x_t2,y_d2)
plt.plot(px,py,color='r')
plt.title('Linear Regression')
plt.xlabel("Price")
plt.ylabel("Year")
plt.show()

# Kilometer linear regression
df3=df.drop(['Name','Location', 'Year','Fuel_Type','Transmission','Owner_Type','Mileage (kmpl)','Engine (CC)','Power (bhp)','Seats'] ,1)
mask_zero=(df3['Kilometers_Driven']==0)
mask_non=(df3['Kilometers_Driven']!=0)
df_zero=df3[mask_zero]
df_non=df3[mask_non]
x_t=df_non['Price']
y_d=df_non['Kilometers_Driven']
x_t1=np.atleast_2d(x_t)
x_t2=np.transpose(x_t1)
y_d1=np.atleast_2d(y_d)
y_d2=np.transpose(y_d1)
reg.fit(x_t2, y_d2)
print(x_t2)
print(y_d2)
px=np.array([x_t2.min()-1,x_t2.max()+1])
py=reg.predict(px[:,np.newaxis])
plt.scatter(x_t2,y_d2)
plt.plot(px,py,color='r')
plt.title('Linear Regression')
plt.xlabel("Price")
plt.ylabel("Kilometers")
plt.show()

# Mileage linear regression
df3=df.drop(['Name','Location', 'Year','Fuel_Type','Transmission','Owner_Type','Kilometers_Driven','Engine (CC)','Power (bhp)','Seats'] ,1)
mask_zero=(df3['Mileage (kmpl)']==0)
mask_non=(df3['Mileage (kmpl)']!=0)
df_zero=df3[mask_zero]
df_non=df3[mask_non]
x_t=df_non['Price']
y_d=df_non['Mileage (kmpl)']
x_t1=np.atleast_2d(x_t)
x_t2=np.transpose(x_t1)
y_d1=np.atleast_2d(y_d)
y_d2=np.transpose(y_d1)
y_d2=y_d2.astype(float)
reg.fit(x_t2, y_d2)
px=np.array([x_t2.min()-1,x_t2.max()+1])
py=reg.predict(px[:,np.newaxis])
plt.scatter(x_t2,y_d2)
plt.plot(px,py,color='r')
plt.title('Linear Regression')
plt.xlabel("Price")
plt.ylabel("Mileage")
plt.show()

# Engine linear regression
df3=df.drop(['Name','Location', 'Year','Fuel_Type','Transmission','Owner_Type','Kilometers_Driven','Mileage (kmpl)','Power (bhp)','Seats'] ,1)
mask_zero=(df3['Engine (CC)']==0)
mask_non=(df3['Engine (CC)']!=0)
df_zero=df3[mask_zero]
df_non=df3[mask_non]
x_t=df_non['Price']
y_d=df_non['Engine (CC)']
x_zerot=df_zero['Price']
x_t1=np.atleast_2d(x_t)
x_t2=np.transpose(x_t1)
y_d1=np.atleast_2d(y_d)
y_d2=np.transpose(y_d1)
y_d2=y_d2.astype(float)
reg.fit(x_t2, y_d2)
px=np.array([x_t2.min()-1,x_t2.max()+1])
py=reg.predict(px[:,np.newaxis])
plt.scatter(x_t2,y_d2)
plt.plot(px,py,color='r')
plt.title('Linear Regression')
plt.xlabel("Price")
plt.ylabel("Engine")
plt.show()

# Power linear regression
df3=df.drop(['Name','Location', 'Year','Fuel_Type','Transmission','Owner_Type','Kilometers_Driven','Mileage (kmpl)','Engine (CC)','Seats'] ,1)
mask_zero=(df3['Power (bhp)']==0)
mask_non=(df3['Power (bhp)']!=0)
df_zero=df3[mask_zero]
df_non=df3[mask_non]
x_t=df_non['Price']
y_d=df_non['Power (bhp)']
x_zerot=df_zero['Price']
x_t1=np.atleast_2d(x_t)
x_t2=np.transpose(x_t1)
y_d1=np.atleast_2d(y_d)
y_d2=np.transpose(y_d1)
y_d2=y_d2.astype(float)
reg.fit(x_t2, y_d2)
px=np.array([x_t2.min()-1,x_t2.max()+1])
py=reg.predict(px[:,np.newaxis])
plt.scatter(x_t2,y_d2)
plt.plot(px,py,color='r')
plt.title('Linear Regression')
plt.xlabel("Price")
plt.ylabel("Power")
plt.show()

