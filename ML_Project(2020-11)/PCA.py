import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/Sample_Data.csv')
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df['Label'] = df['Label'].replace('Benign', 0)
df['Label'] = df['Label'].replace(['Bot', 'DoS', 'DDoS', 'Infilteration', 'BruteForce'], 1)
df['Label'] = df['Label'].replace(np.nan, 2)

df.replace(np.nan, 0, inplace=True)

y = df['Label']
X = df.drop(['Label'], axis=1)

print(X.tail(5))

columns = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1]

print(X.columns)
columns_type = dict(zip(X.columns, columns))

for key, value in columns_type.items():
    if value == 0:
        X[key] = LabelEncoder().fit_transform(X[key])
    else:
        X[key] = StandardScaler().fit_transform(X[key].values.reshape(-1, 1))

cov_mat = np.cov(X.T)
print(np.isnan(cov_mat))
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, 80), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1, 80), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component1', 'principal component2'])

forSaveDf = pd.concat([principalDf, y], axis=1)
forSaveDf.to_csv('newData.csv')
