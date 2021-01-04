import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier
import numpy as np
import warnings
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

# print("pandas version: ", pd.__version__)
# pd.set_option('display.max_row', 500)
# pd.set_option('display.max_columns', 100)

# read dataset
df = pd.read_csv('dataset.csv', encoding='utf-8')
df = df.drop(['Unnamed: 0'], 1)
df = df[(df['Age'] < 40) & (df['Age'] >= 20)]

print(df.isnull().sum())
print(df.head())
print(df.describe())

# dirty value detection
for i in df.columns:
    if len(np.unique(df[i])) > 10:
        continue
    print('{} : {}'.format(i, np.unique(df[i])))

sns.countplot(df['satisfaction'], palette='Paired')
plt.show()

df = df.sample(20000)

# Arrival Delay in Minutes column has NaN values
# fill with mean value
df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].mean(), inplace=True)

# check null value
# print(df.isnull().sum())

# data curation
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
for i in categorical_columns:
    sns.countplot(df[i], palette='Paired')
    plt.show()

# labeling categorical value
label = LabelEncoder()
df['Gender'] = label.fit_transform(df['Gender'].values)
df['Customer Type'] = label.fit_transform(df['Customer Type'].values)
df['Type of Travel'] = label.fit_transform(df['Type of Travel'].values)
df['Class'] = label.fit_transform(df['Class'].values)
df['satisfaction'] = label.fit_transform(df['satisfaction'].values)

# heat map with non-categorical value
heatmap_data = df
colormap = plt.cm.PuBu
plt.figure(figsize=(15, 15))
plt.title("Correlation of Features", y=1.05, size=15)
sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, square=False, cmap=colormap, linecolor="white",
            annot=True, annot_kws={"size": 8})
plt.show()

# feature selection
df.drop(['id', 'Gender', 'Customer Type', 'Age', 'Departure/Arrival time convenient',
         'Ease of Online booking', 'Gate location', 'Food and drink',
        'Departure Delay in Minutes', 'Arrival Delay in Minutes'], 1, inplace=True)

X = df.drop(['satisfaction'], 1)
y = df['satisfaction']

x_columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# use scale
MMScaler = MinMaxScaler()
MMScaler.fit(X)
X = MMScaler.transform(X)

scaled_X = pd.DataFrame(X, columns=x_columns)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 15))

ax1.set_title('Before Scaling')
ax2.set_title('After Min-Max Scaling')

for i in x_columns:
    sns.kdeplot(df[i], ax=ax1)
    sns.kdeplot(scaled_X[i], ax=ax2)

plt.show()

# Random Forest
print('========================= Random Forest ==========================')
parameters = {'max_depth': [1, 10, 100],
              'n_estimators': [1, 10, 100],
              'criterion': ["gini", "entropy"]}

rf = RandomForestClassifier()
kfold = KFold(10, shuffle=True)
rf_model = GridSearchCV(rf, parameters, cv=kfold)
rf_model.fit(X, y)

total_param = rf_model.cv_results_['params']
total_score = rf_model.cv_results_["mean_test_score"]

# for i in range(len(total_score)):
#    print('parameter : ', total_param[i])
#    print('score : ', round(total_score[i], 2))

print('\nBest parameter : ', rf_model.best_params_)
print('Best score : ', round(rf_model.best_score_, 3))

rf_best = rf_model.best_estimator_
rf_score = round(rf_model.best_score_, 3)

# predict y
rf_y_pred = rf_best.predict(X)

# Make confusion matrix
rf_cf = confusion_matrix(y, rf_y_pred)
rf_total = np.sum(rf_cf, axis=1)
rf_cf = rf_cf / rf_total[:, None]
rf_cf = pd.DataFrame(rf_cf, index=["TN", "FN"], columns=["FP", "TP"])

# visualization
plt.figure(figsize=(10, 7))
plt.title("Confusion Matrix with Random Forest")
sns.heatmap(rf_cf, annot=True, annot_kws={"size": 20})
plt.show()

# precision, recall, f1 score
rf_p = round(precision_score(y, rf_y_pred), 3)
print("precision score :", rf_p)
rf_r = round(recall_score(y, rf_y_pred), 3)
print("recall score :", rf_r)
rf_f = round(f1_score(y, rf_y_pred), 3)
print("F1 score :", rf_f)

# Logistic Regression
print('======================= Logistic Regression =======================')
# various parameter

parameters = {'C': [0.1, 1.0, 10.0],
              'solver': ["liblinear", "lbfgs", "sag"],
              'max_iter': [50, 100, 200]}

logisticRegr = LogisticRegression()
lr_model = GridSearchCV(logisticRegr, parameters, cv=kfold)
lr_model.fit(X, y)

total_param = lr_model.cv_results_['params']
total_score = lr_model.cv_results_["mean_test_score"]

# for i in range(len(total_score)):
#    print('parameter : ', total_param[i])
#    print('score : ', round(total_score[i], 2))

print('\nBest parameter : ', lr_model.best_params_)
print('Best score : ', round(lr_model.best_score_, 3))

lr_best = lr_model.best_estimator_
lr_score = round(lr_model.best_score_, 3)

# predict y
lr_y_pred = lr_best.predict(X)

# Make confusion matrix
lr_cf = confusion_matrix(y, lr_y_pred)
lr_total = np.sum(lr_cf, axis=1)
lr_cf = lr_cf / lr_total[:, None]
lr_cf = pd.DataFrame(lr_cf, index=["TN", "FN"], columns=["FP", "TP"])

# visualization
plt.figure(figsize=(10, 7))
plt.title("Confusion Matrix with Logistic Regression")
sns.heatmap(lr_cf, annot=True, annot_kws={"size": 20})
plt.show()

# precision, recall, f1 score
lr_p = round(precision_score(y, lr_y_pred), 3)
print("precision score :", lr_p)
lr_r = round(recall_score(y, lr_y_pred), 3)
print("recall score :", lr_r)
lr_f = round(f1_score(y, lr_y_pred), 3)
print("F1 score :", lr_f)

# SVM
print('=============================== SVM ================================')
# various parameter
parameters = {'C': [0.1, 1.0, 10.0],
               'kernel': ["linear", "rbf", "sigmoid"],
              'gamma': [0.01, 0.1, 1.0, 10.0]}

svclassifier = SVC()
sv_model = GridSearchCV(svclassifier, parameters, cv=kfold)
sv_model.fit(X, y)

total_param = sv_model.cv_results_['params']
total_score = sv_model.cv_results_["mean_test_score"]

# for i in range(len(total_score)):
#    print('parameter : ', total_param[i])
#    print('score : ', round(total_score[i], 2))

print('\nBest parameter : ', sv_model.best_params_)
print('Best score : ', round(sv_model.best_score_, 3))

sv_best = sv_model.best_estimator_
sv_score = round(sv_model.best_score_, 3)

# predict y
sv_y_pred = sv_best.predict(X)

# Make confusion matrix
sv_cf = confusion_matrix(y, sv_y_pred)
sv_total = np.sum(sv_cf, axis=1)
sv_cf = sv_cf / sv_total[:, None]
sv_cf = pd.DataFrame(sv_cf, index=["TN", "FN"], columns=["FP", "TP"])

# visualization
plt.figure(figsize=(10, 7))
plt.title("Confusion Matrix with SVM")
sns.heatmap(sv_cf, annot=True, annot_kws={"size": 20})
plt.show()

# precision, recall, f1 score
sv_p = round(precision_score(y, sv_y_pred), 3)
print("precision score :", sv_p)
sv_r = round(recall_score(y, sv_y_pred), 3)
print("recall score :", sv_r)
sv_f = round(f1_score(y, sv_y_pred), 3)
print("F1 score :", sv_f)

# KNN
print('=============================== KNN ================================')
parameters = {'weights': ['uniform', 'distance'],
              'n_neighbors': [5, 10, 15, 20, 25]}
knn = KNeighborsClassifier()
knn_model = GridSearchCV(knn, parameters, cv=kfold)
knn_model.fit(X, y)

total_param = knn_model.cv_results_['params']
total_score = knn_model.cv_results_["mean_test_score"]

# for i in range(len(total_score)):
#    print('parameter : ', total_param[i])
#    print('score : ', round(total_score[i], 2))

print('\nBest parameter : ', knn_model.best_params_)
print('Best score : ', round(knn_model.best_score_, 3))

knn_best = knn_model.best_estimator_
knn_score = round(knn_model.best_score_, 3)

# predict y
knn_y_pred = knn_best.predict(X)

# Make confusion matrix
knn_cf = confusion_matrix(y, knn_y_pred)
knn_total = np.sum(knn_cf, axis=1)
knn_cf = knn_cf / knn_total[:, None]
knn_cf = pd.DataFrame(knn_cf, index=["TN", "FN"], columns=["FP", "TP"])

# visualization
plt.figure(figsize=(10, 7))
plt.title("Confusion Matrix with KNN")
sns.heatmap(knn_cf, annot=True, annot_kws={"size": 20})
plt.show()

# precision, recall, f1 score
knn_p = round(precision_score(y, knn_y_pred), 3)
print("precision score :", knn_p)
knn_r = round(recall_score(y, knn_y_pred), 3)
print("recall score :", knn_r)
knn_f = round(f1_score(y, knn_y_pred), 3)
print("F1 score :", knn_f)

# GradientBoostingClassifier
print('================= Gradient Boosting Classifier ======================')
parameters = {"n_estimators": range(50, 100, 25),
              "max_depth": [1, 2, 4],
              "learning_rate": [0.0001, 0.001, 0.01, 0.1],
              "subsample": [0.7, 0.9]}

gb = GradientBoostingClassifier(random_state=2020)
gb_model = GridSearchCV(gb, parameters, cv=kfold)
gb_model.fit(X, y)

total_param = knn_model.cv_results_['params']
total_score = knn_model.cv_results_["mean_test_score"]

# for i in range(len(total_score)):
#    print('parameter : ', total_param[i])
#    print('score : ', round(total_score[i], 2))

print('\nBest parameter : ', gb_model.best_params_)
print('Best score : ', round(gb_model.best_score_, 3))

gb_best = gb_model.best_estimator_
gb_score = round(gb_model.best_score_, 3)

# predict y
gb_y_pred = gb_best.predict(X)

# Make confusion matrix
gb_cf = confusion_matrix(y, gb_y_pred)
gb_total = np.sum(gb_cf, axis=1)
gb_cf = gb_cf / gb_total[:, None]
gb_cf = pd.DataFrame(gb_cf, index=["TN", "FN"], columns=["FP", "TP"])

# visualization
plt.figure(figsize=(10, 7))
plt.title("Confusion Matrix with Gradient Boosting")
sns.heatmap(gb_cf, annot=True, annot_kws={"size": 20})
plt.show()

# precision, recall, f1 score
gb_p = round(precision_score(y, gb_y_pred), 3)
print("precision score :", gb_p)
gb_r = round(recall_score(y, gb_y_pred), 3)
print("recall score :", gb_r)
gb_f = round(f1_score(y, gb_y_pred), 3)
print("F1 score :", gb_f)

# XGBClassifier
print('========================= XGB Classifier ==========================')
parameters = {'booster': ['gbtree'],
              'max_depth': [5, 6, 8],
              'min_child_weight': [1, 3, 5],
              'gamma': [0, 1, 2, 3]}
xgb = XGBClassifier()
xgb_model = GridSearchCV(xgb, parameters, cv=kfold)
xgb_model.fit(X, y)

total_param = xgb_model.cv_results_['params']
total_score = xgb_model.cv_results_["mean_test_score"]

# for i in range(len(total_score)):
#    print('parameter : ', total_param[i])
#    print('score : ', round(total_score[i], 2))

print('\nBest parameter : ', xgb_model.best_params_)
print('Best score : ', round(xgb_model.best_score_, 3))
xgb_best = xgb_model.best_estimator_
xgb_score = round(xgb_model.best_score_, 3)

# predict y
xgb_y_pred = xgb_best.predict(X)

# Make confusion matrix
xgb_cf = confusion_matrix(y, xgb_y_pred)
xgb_total = np.sum(xgb_cf, axis=1)
xgb_cf = xgb_cf / xgb_total[:, None]
xgb_cf = pd.DataFrame(xgb_cf, index=["TN", "FN"], columns=["FP", "TP"])

# visualization
plt.figure(figsize=(10, 7))
plt.title("Confusion Matrix with XGB")
sns.heatmap(xgb_cf, annot=True, annot_kws={"size": 20})
plt.show()

# precision, recall, f1 score
xgb_p = round(precision_score(y, xgb_y_pred), 3)
print("precision score :", xgb_p)
xgb_r = round(recall_score(y, xgb_y_pred), 3)
print("recall score :", xgb_r)
xgb_f = round(f1_score(y, xgb_y_pred), 3)
print("F1 score :", xgb_f)

print('============================== Result ==============================\n')
print('Random Forest score : ', rf_score)
print('Logistic Regression score : ', lr_score)
print('SVM score : ', sv_score)
print('KNN score : ', knn_score)
print('Gradient Boosting Classifier score : ', gb_score)
print('XGB Classifier score : ', xgb_score)

# Voting
print('========================= Voting Classifier ========================\n')
voting = VotingClassifier(estimators=[('rfc', rf_best),
                                      ('knn', knn_best),
                                      ('svc', sv_best),
                                      ('gbc', gb_best),
                                      ('xgb', xgb_best)],
                          voting='hard', n_jobs=5)

scores = cross_val_score(voting, X, y, cv=10)
print("Voting classifier score : {:.2f}".format(scores.mean()))


def rocvis(true, prob, label):
    fpr, tpr, thresholds = roc_curve(true, prob)
    plt.plot(fpr, tpr, linestyle='--', label=label)


fig, ax = plt.subplots(figsize=(20, 10))
plt.plot([0, 1], [0, 1], linestyle='--')
rocvis(y, rf_y_pred, "Random Forest")
rocvis(y, lr_y_pred, "Logistic Regression")
rocvis(y, sv_y_pred, "SVM")
rocvis(y, knn_y_pred, "KNN")
rocvis(y, gb_y_pred, "Gradient Boosting")
rocvis(y, xgb_y_pred, "XGB")
plt.legend(fontsize=18)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Models Roc Curve", fontsize=25)
plt.show()

for test_case in range(1, T + 1):
    max = 0
    for x in range(10) :
        b = int(input())
        if (b>max) :
        	max = b
    	print("#{} {}".format(T,max))
