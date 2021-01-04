import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sn

dataset = pd.read_csv('Sample_Data.csv', header=None)
dataset.drop(0, axis=1, inplace=True)
dataset.drop(dataset.index[0], inplace=True)

# Preprocessing to classification
dataset.drop([3], axis=1, inplace=True)
dataset[80] = dataset[80].replace('Benign', 0)
dataset[80] = dataset[80].replace('Bot', 1)
dataset[80] = dataset[80].replace('DoS', 1)
dataset[80] = dataset[80].replace('DDoS', 1)
dataset[80] = dataset[80].replace('Infilteration', 1)
dataset[80] = dataset[80].replace('BruteForce', 1)
cols = dataset.columns[dataset.dtypes.eq(object)]
dataset[cols] = dataset[cols].apply(pd.to_numeric, errors='coerce')

# print(dataset.isnull().sum())

# unlabeld label data
unlabeled = dataset[dataset[80].isnull()]
unlabeled[80] = 1

# labeld data
dataset = dataset.dropna(subset=[80])

# split dataset
dataset = dataset[~dataset.isin([np.nan, np.inf, -np.inf]).any(1)].astype(np.float32)

train, test = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=5000)

X_train = train.iloc[:, 0:78]
y_train = train[80]
X_test = test.iloc[:, 0:78]
y_test = test[80]
# print(test)
# print(y_test)

model = DecisionTreeClassifier()
model = model.fit(X_train, y_train)

pre = model.predict(X_test)
print('labeled Accuracy: %.2f' % accuracy_score(y_test, pre))
confusion_matrix = pd.crosstab(y_test, pre, rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True)
plt.title('Labeled data confusion matrix', fontsize=20)

plt.show()

unpredict = (y_test != pre)
# print("Labeled data (Test != Unpredicted) ")
# print(unpredict.value_counts())


X = unlabeled.iloc[:, 0:78]
Y = unlabeled[80]
Y.fillna(1)

pre = model.predict(X)
print('Unlabeled Accuracy: %.2f' % accuracy_score(Y, pre))

confusion_matrix = pd.crosstab(Y, pre, rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True)
plt.title('Unlabeled data confusion matrix', fontsize=20)
plt.show()

# sn.heatmap(confusion_matrix,annot=True)
# plt.title('Unlabeled data confusion matrix', fontsize=20)
# plt.show()

unlabeled_unpredict = (Y != pre)
# print("Unlabeled data (Test != Unpredicted) ")
# print(unlabeled_unpredict.value_counts())

unlabeled_unpredict = (unlabeled[80] != pre)
unlabeled[80] = ''

# print(unlabeled)
undetected_data = unlabeled[unlabeled_unpredict];
undetected_data = pd.concat([undetected_data, test[unpredict]])

print("Undetected data")
print(undetected_data)
undetected_data.to_csv('Sample_Data_after.csv', index=False, encoding='cp949')
