from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn import svm

from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def scoring(estimator, x, y):
    y_pred = estimator.predict(x)
    # y_test = y >= 5
    # y_pred = y_pred >= 5
    return f1_score(y, y_pred, labels=[False, True], average='micro')


def main():
    train_features = pd.read_csv('./data/NKI/train_features.csv', index_col=0)
    test_features = pd.read_csv('./data/NKI/test_features.csv', index_col=0)
    nki_survival = pd.read_csv('./data/NKI/nki_survival.csv', index_col=0)
    survival_data = nki_survival.loc[:, ['Survival_2005']]
    survival_data.rename(columns={'Survival_2005': 'Survival_2005_Float'}, inplace=True)
    test_features = test_features.dropna(axis=0)

    train_features = train_features.loc[train_features.index.difference(test_features.index)]
    concat = pd.concat([train_features, test_features], axis=0)

    survival_data = survival_data.loc[survival_data.index.intersection(concat.index)]
    concat = pd.concat([survival_data, concat], axis=1)

    x = concat.iloc[:, 2:]
    y = concat.iloc[:, 1]

    # ros = RandomOverSampler(random_state=0)
    # x_resampled, y_resampled = ros.fit_sample(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=5)

    train_survival = survival_data.loc[survival_data.index.intersection(train_features.index)]
    test_survival = survival_data.loc[survival_data.index.intersection(test_features.index)]

    train_features = pd.concat([train_survival, train_features], axis=1)
    test_features = pd.concat([test_survival, test_features], axis=1)

    x_train = train_features.iloc[:, 2:]
    y_train = train_features.iloc[:, 1]

    x_test = test_features.iloc[:, 2:]
    y_test = test_features.iloc[:, 1]

    # clf = svm.SVC(C=0.1, kernel='rbf', gamma=0.00001, decision_function_shape='ovo')
    # clf.fit(x_train, y_train.ravel())
    # print(clf.score(x_train, y_train))

    # y_test_hat = clf.predict(x_test)
    # print(classification_report(y_test, y_test_hat))

    model = LogisticRegression(random_state=0, solver='lbfgs', )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)

    for p in y_pred_proba:
        print("%6.2f" % (p[0] * 100) + "\t" + "%6.2f" % (p[1] * 100))
    # y_test = y_test >= 5
    # y_pred = y_pred >= 5
    print(classification_report(y_test, y_pred))
    accs = cross_val_score(model, x, y=y, scoring=scoring, cv=10, n_jobs=1)
    print(accs)
    print(np.mean(accs))
    pass


if __name__ == '__main__':
    main()
