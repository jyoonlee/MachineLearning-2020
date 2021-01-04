import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Read data
data = pd.read_csv('Indicators.csv')

column = ['CountryCode', "GDP", "Inflation", "Exports", "Imports",
          "Population", "Manufacturing", "Industry", "Agriculture", "Year"]
indicator = ['GDP per capita (current US$)', 'Inflation, GDP deflator (annual %)',
             'Exports of goods and services (% of GDP)', 'Imports of goods and services (% of GDP)',
             'Population growth (annual %)', 'Manufacturing, value added (% of GDP)',
             'Industry, value added (% of GDP)', 'Agriculture, value added (% of GDP)']
newColumn = column[1:9]

indices = data['CountryCode'].value_counts().index
indices = indices.sort_values().tolist()
length = len(indices)
indices = list(set(indices))
indices = sorted(indices)

data = data[((data['IndicatorCode'] == 'NY.GDP.PCAP.CD') |
            (data['IndicatorCode'] == 'NY.GDP.DEFL.KD.ZG') |
            (data['IndicatorCode'] == 'NE.EXP.GNFS.ZS') |
            (data['IndicatorCode'] == 'NE.IMP.GNFS.ZS') |
            (data['IndicatorCode'] == 'SP.POP.GROW') |
            (data['IndicatorCode'] == 'NV.IND.MANF.ZS') |
            (data['IndicatorCode'] == 'NV.IND.TOTL.ZS') |
            (data['IndicatorCode'] == 'NV.AGR.TOTL.ZS'))]
data = data[data.Year > 1990]
data = data.sort_values(by='CountryCode')

dfIdx = 0
df = pd.DataFrame(columns=column, index=range(len(data)))
for i in range(len(data)):
    for j in range(len(indicator)):
        if data.iloc[i, 2] == indicator[j]:
            df.iloc[dfIdx, 0] = data.iloc[i, 1]
            df.iloc[dfIdx, 9] = data.iloc[i, 4]
            df.iloc[dfIdx, j+1] = data.iloc[i, 5]
    dfIdx += 1

k = -1
temp = []
tempCC = ''
tempYear = 1990
tempIdx = 0
new = pd.DataFrame(index=indices, columns=newColumn)

for i in range(len(df)):
    if df.iloc[i, 0] != tempCC:
        tempCC = df.iloc[i, 0]
        tempIdx = i
        k += 1
    else:
        for j in range(len(newColumn)):
            tempYear = df.iloc[i, 9]
            new.iloc[k, j] = df.iloc[tempIdx:i+1, j+1].max(skipna=True)

new = new.dropna()

# Scaling
x = StandardScaler().fit_transform(new)
# x = MinMaxScaler().fit_transform(new)
# x = RobustScaler().fit_transform(new)
# x = Normalizer().fit_transform(new)

# PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
newX = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])



# DBSCAN

# Parameters of DBSCAN
eps = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
min_samples = [0.1, 1, 3, 5, 10, 15, 20, 30, 50, 100]

for i in range(np.size(eps)):
    for j in range(np.size(min_samples)):
        dbscan = DBSCAN(eps=eps[i], min_samples=min_samples[j])
        clusters_d = dbscan.fit_predict(newX)

        # show scatter
        plt.scatter(newX.iloc[:, 0], newX.iloc[:, 1], c=clusters_d, alpha=0.7)
        plt.title("DBSCAN with eps = %f, min_sample = %d" % (eps[i], min_samples[j]))
        plt.savefig('./plot/dbscan'+str(eps[i])+str(min_samples[j])+'.png', dpi=300)


# K-Means

# Parameters of K-Means
n_clusters = [4]
max_iter_k = [1, 10, 50, 100, 200, 300]

for i in range(np.size(n_clusters)):
    for j in range(np.size(max_iter_k)):
        k_means = KMeans(n_clusters=n_clusters[i], max_iter=max_iter_k[j], init='k-means++')
        clusters_k = k_means.fit_predict(newX)
        # show scatter
        plt.scatter(newX.iloc[:, 0], newX.iloc[:, 1], c=clusters_k, alpha=0.5)
        plt.title("K-Means with n_clusters = %d, max_iter = %d" % (n_clusters[i], max_iter_k[j]))
        plt.savefig('./plot/kmeans'+str(n_clusters[i])+str(max_iter_k[j])+'.png', dpi=300)


# EM algorithms

# Parameters of EM
n_components = [3, 4]
max_iter_em = [10, 30, 50, 100, 200, 300]

for i in range(np.size(n_components)):
    for j in range(np.size(max_iter_em)):
        EM = GaussianMixture(n_components=n_components[i], max_iter=max_iter_em[j])
        clusters_em = EM.fit_predict(newX)

        # show scatter
        plt.scatter(newX.iloc[:, 0], newX.iloc[:, 1], c=clusters_em, alpha=0.5)
        plt.title("EM with n_components = %d, max_iter = %d" % (n_components[i], max_iter_em[j]))
        plt.savefig('./plot/em'+str(n_components[i])+str(max_iter_em[j])+'.png', dpi=300)


# Income Group in Country
country = pd.read_csv('Country.csv')
temp = country
temp.drop(temp.iloc[:, 1:8], axis=1, inplace=True)
temp.drop(temp.iloc[:, 2:], axis=1, inplace=True)
for i in range(len(temp['IncomeGroup'])):
    if temp['IncomeGroup'][i] == 'High income: nonOECD':
        temp['IncomeGroup'][i] = 'High income'
    if temp['IncomeGroup'][i] == 'High income: OECD':
        temp['IncomeGroup'][i] = 'High income'
print(temp)
high = 0
up_middle = 0
low_middle = 0
low = 0
for i in range(len(temp['IncomeGroup'])):
    if temp['IncomeGroup'][i] == 'High income':
        high += 1
    elif temp['IncomeGroup'][i] == 'Upper middle income':
        up_middle += 1
    elif temp['IncomeGroup'][i] == 'Lower middle income':
        low_middle += 1
    else:
        low += 1

print('High income: %d\nUpper middle incomd: %d\nLower middle income: %d\nLow income: %d' % (high, up_middle, low_middle, low))