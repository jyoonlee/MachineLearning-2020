import pandas as pd
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, GridSearchCV, train_test_split

df = pd.read_csv('preprocessing_data.csv', encoding='utf-8')

# Split
X = df.drop(['Price'], 1)
y = df['Price']

# test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
knn = KNeighborsRegressor(n_neighbors=8)
knn.fit(X_train, y_train)
print('score : %.2f\n\n' % knn.score(X_test, y_test))

# k-fold
kfold = KFold(5, shuffle=True)
parameters = {'n_neighbors': range(3, 10)}
clf = GridSearchCV(KNeighborsRegressor(), parameters, cv=kfold)
clf.fit(X, y)

print('best k-value: ', clf.best_params_)
print('best score: %.2f' % clf.best_score_)

# ensemble learning
model = BaggingRegressor(KNeighborsRegressor(), n_estimators=10, max_samples=50, bootstrap=True)
model.fit(X_train, y_train)
model.predict(X_test)
print('k = {} accuracy : %.2f'.format(10) % model.score(X_test, y_test))

# ensemble learning
model = BaggingRegressor(KNeighborsRegressor(), n_estimators=50, max_samples=50, bootstrap=True)
model.fit(X_train, y_train)
model.predict(X_test)
print('k = {} accuracy : %.2f'.format(50) % model.score(X_test, y_test))

# ensemble learning
model = BaggingRegressor(KNeighborsRegressor(), n_estimators=100, max_samples=50, bootstrap=True)
model.fit(X_train, y_train)
model.predict(X_test)
print('k = {} accuracy : %.2f'.format(100) % model.score(X_test, y_test))

# ensemble learning
model = BaggingRegressor(KNeighborsRegressor(), n_estimators=200, max_samples=50, bootstrap=True)
model.fit(X_train, y_train)
model.predict(X_test)
print('k = {} accuracy : %.2f'.format(200) % model.score(X_test, y_test))

# ensemble learning
model = BaggingRegressor(KNeighborsRegressor(), n_estimators=300, max_samples=50, bootstrap=True)
model.fit(X_train, y_train)
model.predict(X_test)
print('k = {} accuracy : %.2f'.format(300) % model.score(X_test, y_test))
