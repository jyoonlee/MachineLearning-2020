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
# change value DirtydData to 0
df.replace('null ', 0, inplace=True)
df.replace('?', 0, inplace=True)
reg=LinearRegression()

# outlier
df['Kilometers_Driven']=df['Kilometers_Driven'].astype(float)
# detect kilometer more than 1000,000
df['Kilometers_Driven'] = df['Kilometers_Driven'].where(df['Kilometers_Driven'].between(0, 1000000))
df = df.fillna(0)

# name, location
df2=df.drop(['Year', 'Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Mileage (kmpl)',
             'Engine (CC)','Power (bhp)','Seats','Price'] ,1)

# +Year column
df3=df.drop(['Name','Location', 'Kilometers_Driven','Fuel_Type','Transmission',
             'Owner_Type','Mileage (kmpl)','Engine (CC)','Power (bhp)','Seats'] ,1)
# divide zero or not.
mask_zero=(df3['Year']==0)
mask_non=(df3['Year']!=0)
df_zero=df3[mask_zero]
df_non=df3[mask_non]
x_t=df_non['Price']
y_d=df_non['Year']
x_zerot=df_zero['Price']
x_t1=np.atleast_2d(x_t)
# transpose price
x_t2=np.transpose(x_t1)
y_d1=np.atleast_2d(y_d)
# transepose year
y_d2=np.transpose(y_d1)
x_zerot1=np.atleast_2d(x_zerot)
x_zerot2=np.transpose(x_zerot1)
reg.fit(x_t2, y_d2)
y_predict=reg.predict(x_zerot2)
y_predict1=np.transpose(y_predict)
# squeeze size 1
y_predict2=np.squeeze(y_predict1)
df_temp1=pd.DataFrame(y_predict2)
df_temp2=df_zero['Price']
# predicted year to 'new' column
df_zero['new']=y_predict2
df_zero.drop(['Year','Price'],axis='columns',inplace=True)
# match with price with existing line number
df_need=pd.concat([df_zero,df_temp2],axis=1,ignore_index=True)
df_need.columns=(['Year','Price'])
# concat with existing dataset
df_final=pd.concat([df_need,df_non],axis=0,ignore_index=False)
# sort with line number
df_final2=df_final.sort_index()
df_final2.drop(['Price'],axis='columns', inplace=True)
# new year column concat with df2
df2=pd.concat([df2,df_final2],axis=1,ignore_index=False)

# +Kilometer
df3=df.drop(['Name','Location', 'Year','Fuel_Type','Transmission','Owner_Type','Mileage (kmpl)','Engine (CC)','Power (bhp)','Seats'] ,1)
mask_zero=(df3['Kilometers_Driven']==0)
mask_non=(df3['Kilometers_Driven']!=0)
df_zero=df3[mask_zero]
df_non=df3[mask_non]
x_t=df_non['Price']
y_d=df_non['Kilometers_Driven']
x_zerot=df_zero['Price']
x_t1=np.atleast_2d(x_t)
x_t2=np.transpose(x_t1)
y_d1=np.atleast_2d(y_d)
y_d2=np.transpose(y_d1)
x_zerot1=np.atleast_2d(x_zerot)
x_zerot2=np.transpose(x_zerot1)
reg.fit(x_t2, y_d2)
y_predict=reg.predict(x_zerot2)
y_predict1=np.transpose(y_predict)
y_predict2=np.squeeze(y_predict1)
df_temp1=pd.DataFrame(y_predict2)
df_temp2=df_zero['Price']
df_zero['new']=y_predict2
df_zero.drop(['Kilometers_Driven','Price'],axis='columns',inplace=True)
df_need=pd.concat([df_zero,df_temp2],axis=1,ignore_index=True)
df_need.columns=(['Kilometers_Driven','Price'])
df_final=pd.concat([df_need,df_non],axis=0,ignore_index=False)
df_final2=df_final.sort_index()
df_final2.drop(['Price'],axis='columns', inplace=True)
df2=pd.concat([df2,df_final2],axis=1,ignore_index=False)

# +Fuel with bfill
df3=df.drop(['Name','Location', 'Year','Kilometers_Driven','Transmission','Owner_Type','Mileage (kmpl)','Engine (CC)','Power (bhp)','Seats','Price'] ,1)
df3.replace(0, np.nan, inplace=True)
df3.fillna(method='bfill', inplace=True)
df2=pd.concat([df2,df3],axis=1,ignore_index=False)

# +Trans with bfill
df3=df.drop(['Name','Location', 'Year','Kilometers_Driven','Fuel_Type','Owner_Type','Mileage (kmpl)','Engine (CC)','Power (bhp)','Seats','Price'] ,1)
df3.replace(0, np.nan, inplace=True)
df3.fillna(method='bfill', inplace=True)
df2=pd.concat([df2,df3],axis=1,ignore_index=False)

# +Owner with bfill
df3=df.drop(['Name','Location', 'Year','Kilometers_Driven','Fuel_Type','Transmission','Mileage (kmpl)','Engine (CC)','Power (bhp)','Seats','Price'] ,1)
df3.replace(0, np.nan, inplace=True)
df3.fillna(method='bfill', inplace=True)
df2=pd.concat([df2,df3],axis=1,ignore_index=False)

# +Mileage
df3=df.drop(['Name','Location', 'Year','Fuel_Type','Transmission','Owner_Type','Kilometers_Driven','Engine (CC)','Power (bhp)','Seats'] ,1)
mask_zero=(df3['Mileage (kmpl)']==0)
mask_non=(df3['Mileage (kmpl)']!=0)
df_zero=df3[mask_zero]
df_non=df3[mask_non]
x_t=df_non['Price']
y_d=df_non['Mileage (kmpl)']
x_zerot=df_zero['Price']
x_t1=np.atleast_2d(x_t)
x_t2=np.transpose(x_t1)
y_d1=np.atleast_2d(y_d)
y_d2=np.transpose(y_d1)
x_zerot1=np.atleast_2d(x_zerot)
x_zerot2=np.transpose(x_zerot1)
reg.fit(x_t2, y_d2)
y_predict=reg.predict(x_zerot2)
y_predict1=np.transpose(y_predict)
y_predict2=np.squeeze(y_predict1)
df_temp1=pd.DataFrame(y_predict2)
df_temp2=df_zero['Price']
df_zero['new']=y_predict2
df_zero.drop(['Mileage (kmpl)','Price'],axis='columns',inplace=True)
df_need=pd.concat([df_zero,df_temp2],axis=1,ignore_index=True)
df_need.columns=(['Mileage (kmpl)','Price'])
df_final=pd.concat([df_need,df_non],axis=0,ignore_index=False)
df_final2=df_final.sort_index()
df_final2.drop(['Price'],axis='columns', inplace=True)
df2=pd.concat([df2,df_final2],axis=1,ignore_index=False)

# +Engine with bfill
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
x_zerot1=np.atleast_2d(x_zerot)
x_zerot2=np.transpose(x_zerot1)
reg.fit(x_t2, y_d2)
y_predict=reg.predict(x_zerot2)
y_predict1=np.transpose(y_predict)
y_predict2=np.squeeze(y_predict1)
df_temp1=pd.DataFrame(y_predict2)
df_temp2=df_zero['Price']
df_zero['new']=y_predict2
df_zero.drop(['Engine (CC)','Price'],axis='columns',inplace=True)
df_need=pd.concat([df_zero,df_temp2],axis=1,ignore_index=True)
df_need.columns=(['Engine (CC)','Price'])
df_final=pd.concat([df_need,df_non],axis=0,ignore_index=False)
df_final2=df_final.sort_index()
df_final2.drop(['Price'],axis='columns', inplace=True)
df2=pd.concat([df2,df_final2],axis=1,ignore_index=False)

# +Power
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
x_zerot1=np.atleast_2d(x_zerot)
x_zerot2=np.transpose(x_zerot1)
reg.fit(x_t2, y_d2)
y_predict=reg.predict(x_zerot2)
y_predict1=np.transpose(y_predict)
y_predict2=np.squeeze(y_predict1)
df_temp1=pd.DataFrame(y_predict2)
df_temp2=df_zero['Price']
df_zero['new']=y_predict2
df_zero.drop(['Power (bhp)','Price'],axis='columns',inplace=True)
df_need=pd.concat([df_zero,df_temp2],axis=1,ignore_index=True)
df_need.columns=(['Power (bhp)','Price'])
df_final=pd.concat([df_need,df_non],axis=0,ignore_index=False)
df_final2=df_final.sort_index()
df_final2.drop(['Price'],axis='columns', inplace=True)
df2=pd.concat([df2,df_final2],axis=1,ignore_index=False)

# +Seats with bfill
df3=df.drop(['Name','Location', 'Year','Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Mileage (kmpl)','Engine (CC)','Power (bhp)','Price'] ,1)
df3.replace(0, np.nan, inplace=True)
df3.fillna(method='bfill', inplace=True)
df2=pd.concat([df2,df3],axis=1,ignore_index=False)

# +Price
df3=df.drop(['Name','Location', 'Year','Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Mileage (kmpl)','Engine (CC)','Power (bhp)','Seats'] ,1)
df2=pd.concat([df2,df3],axis=1,ignore_index=False)

print(df2)

# labeling categorical value
label = LabelEncoder()
df2['Name'] = label.fit_transform(df2['Name'].values)
df2['Location'] = label.fit_transform(df2['Location'].values)
df2['Owner_Type'] = label.fit_transform(df2['Owner_Type'].values)
df2['Fuel_Type'] = label.fit_transform(df2['Fuel_Type'].values)
df2['Transmission'] = label.fit_transform(df2['Transmission'].values)
print(df2)

# heat map with non-categorical value
heatmap_data = df2[['Year', 'Kilometers_Driven', 'Mileage (kmpl)', 'Engine (CC)', 'Power (bhp)', 'Seats', 'Price']]
colormap = plt.cm.PuBu
plt.figure(figsize=(15, 15))
plt.title("Correlation of Features", y=1.05, size=15)
sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, square=False, cmap=colormap, linecolor="white",
            annot=True, annot_kws={"size": 16})

plt.show()
print(heatmap_data)


#df2.drop(['Location', 'Fuel_Type', 'Kilometers_Driven', 'Mileage (kmpl)', 'Price', 'Seats',], 1, inplace=True)
df2.drop(['Location', 'Fuel_Type', 'Kilometers_Driven', 'Mileage (kmpl)', 'Seats',], 1, inplace=True)
df2.to_csv("preprocessing_data.csv")
