from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import GridSearchCV, KFold

warnings.filterwarnings('ignore')

# print("pandas version: ", pd.__version__)
pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)

# extract benign data from original data
ori_df = pd.read_csv('Sample_Data.csv', encoding='utf-8', header=None)
ori_df.drop(0, axis=1, inplace=True)
ori_df.drop(ori_df.index[0], inplace=True)
ori_df.drop([3], axis=1, inplace=True)
ori_df[80] = ori_df[80].replace('Benign', 0)
ori_df = ori_df.loc[ori_df[80] == 0]
ori_df[80] = ori_df[80].replace(0, 1)
cols = ori_df.columns[ori_df.dtypes.eq(object)]
ori_df[cols] = ori_df[cols].apply(pd.to_numeric, errors='coerce')
columns = ori_df.columns

# extract unlabeled data not detected
df = pd.read_csv('Sample_Data_after_final.csv', encoding='utf-8')
print(df.groupby('80').size())
df.columns = columns
df.drop(df.loc[df[80] == 1].index, inplace=True)  # undetected infilteration data
# df[80] = df[80].replace(1, -1)
df.drop(df.loc[df[80] == 0].index, inplace=True)  # undetected benign data
# df[80].dropna()
df[80].fillna(-1, inplace=True)
anomaly_data = len(df)

ori_df = ori_df[~ori_df.isin([np.nan, np.inf, -np.inf]).any(1)].astype(np.float32)
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].astype(np.float32)

train_data = ori_df.drop([80], 1)
test_data = df.drop([80], 1)
y = df[80]


def scorer_f(estimator, X):  # your own scorer
    return np.mean(estimator.score_samples(X))


clf = IsolationForest(bootstrap=False)

n_value = [100, 200, 300]
c_value = [0.01, 0.05, 0.1]

num = 1
max_value = 0

for i in n_value:
    for j in c_value:
        print('{}. n_estimators: {}, contamination: {}'.format(num, i, j))
        clf = IsolationForest(n_estimators=i, max_samples=300, contamination=j, max_features=1.0, bootstrap=False)
        clf.fit(train_data)
        pred = clf.predict(test_data)
        score = accuracy_score(y, pred)

        print('Anomaly detection accuracy: {}'.format(round(score,2)))

        if score > max_value:
            max_value = score
            max_pred = pred

        num += 1

test_data['anomaly'] = max_pred
print(test_data['anomaly'].value_counts())

# confusion matrix
matrix = confusion_matrix(y, max_pred)
print(matrix)

# print('Anomaly detection accuracy: %.2f' % accuracy_score(y, pred))

confusion_matrix = pd.crosstab(y, max_pred, rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True)
plt.title('Anomaly detection confusion matrix', fontsize=20)
plt.show()
