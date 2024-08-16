import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz
import sys
from threading import Thread
import os
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.sparse import dok_matrix
import os
import subprocess
import numpy as np
import pandas as pd
import sys
import joblib
from joblib import parallel_config
from tqdm import tqdm
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, make_scorer
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


def format_time(t1, t2):
    elapsed_time = t2 - t1
    if elapsed_time < 60:
        return f"{elapsed_time:.2f} seconds"
    elif elapsed_time < 3600:
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60
        return f"{minutes:.0f} minutes {seconds:.2f} seconds"
    elif elapsed_time < 86400:
        hours = elapsed_time // 3600
        remainder = elapsed_time % 3600
        minutes = remainder // 60
        seconds = remainder % 60
        return f"{hours:.0f} hours {minutes:.0f} minutes {seconds:.2f} seconds"
    else:
        days = elapsed_time // 86400
        remainder = elapsed_time % 86400
        hours = remainder // 3600
        remainder = remainder % 3600
        minutes = remainder // 60
        seconds = remainder % 60
        return f"{days:.0f} days {hours:.0f} hours {minutes:.0f} minutes {seconds:.2f} seconds"



dataset_path = r"/home/23m1521/ashish/Kaggle/_3_IE506_2024_Programming_Challenge/dataset"
train_txt_path  = r"/home/23m1521/ashish/Kaggle/_3_IE506_2024_Programming_Challenge/dataset/IE506_2024_progchallenge_train.txt"
test_txt_path = r"/home/23m1521/ashish/Kaggle/_3_IE506_2024_Programming_Challenge/dataset/IE506_2024_progchallenge_test.txt"
load_features_path = r"/home/23m1521/ashish/Kaggle/_3_IE506_2024_Programming_Challenge/dataset/csr_feature.npz"
load_labels_path = r"/home/23m1521/ashish/Kaggle/_3_IE506_2024_Programming_Challenge/dataset/csr_label.npz"
load_cols_F_path = r"/home/23m1521/ashish/Kaggle/_3_IE506_2024_Programming_Challenge/dataset/cols_F.joblib"
load_cols_C_path = r"/home/23m1521/ashish/Kaggle/_3_IE506_2024_Programming_Challenge/dataset/cols_C.joblib"
load_features_sub_path = r"/home/23m1521/ashish/Kaggle/_3_IE506_2024_Programming_Challenge/dataset/csr_feature_sub.npz"
model_save_path = r"/home/23m1521/ashish/Kaggle/_3_IE506_2024_Programming_Challenge/models"


features_arr = load_npz(load_features_path)
X_sub = load_npz(load_features_sub_path)
labels_arr = load_npz(load_labels_path)
cols_F = joblib.load(load_cols_F_path)
cols_C = joblib.load(load_cols_C_path)


###---------------------------------------\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
###----------------- L2LR ----------------\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
###---------------------------------------\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
t1 = timer()
TRIAL_NAME = "L2LR"

### GridSearch Cross-Validation \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
print(f"Doing {TRIAL_NAME} ClassifierChain GridSearch Cross-Validation....")

CV = 10
C_values = [0.001, 0.01, 0.1, 1, 10]
Order = [cols_C.index(i) for i in sorted(cols_C, key=lambda x: int(x.split('_')[1]))]
param_grid = [{'C':C_values}]


base_clf = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000, random_state=42, tol=1e-5, n_jobs=-1)
grid = GridSearchCV(estimator = base_clf,
                    param_grid=param_grid,
                    scoring={'f1': make_scorer(f1_score, average='micro')},
                    refit='f1',
                    cv=CV, verbose=5,
                    n_jobs=-1)

CCclf = ClassifierChain(grid, order=Order, cv=10, random_state=42, verbose=True)
CCclf.fit(features_arr, np.squeeze(labels_arr.toarray()))

joblib.dump(CCclf, f"{model_save_path}/MOCCGS_{TRIAL_NAME}.joblib")
print("Model saved_____________________________")
print(f'Time taken: {format_time(t1, timer())}')

X_train, X_test, y_train_full, y_test_full = train_test_split(features_arr, labels_arr, test_size=0.2, random_state=42)
y_train_full = np.squeeze(y_train_full.toarray())
y_test_full = np.squeeze(y_test_full.toarray())

train_pred = CCclf.predict(X_train)
test_pred = CCclf.predict(X_test)
train_acc = 100*np.sum(np.sum((train_pred == y_train_full), axis=1)/train_pred.shape[1])/train_pred.shape[0]
test_acc = 100*np.sum(np.sum((test_pred == y_test_full), axis=1)/test_pred.shape[1])/test_pred.shape[0]
print(f"Train acc: {train_acc} %\nTest acc: {test_acc} %")


print(str("train_acc").ljust(20), str("test_acc").ljust(20), str("train_f1").ljust(20), str("test_f1").rjust(20))
for i in range(train_pred.shape[1]):
    train_acc = 100*accuracy_score(y_train_full[:,i], train_pred[:, i])
    test_acc = 100*accuracy_score(y_test_full[:,i], test_pred[:, i])
    train_f1 = 100*f1_score(y_train_full[:,i], train_pred[:, i], average='micro')
    test_f1 = 100*f1_score(y_test_full[:,i], test_pred[:, i], average='micro')
    if i<9:
        print(f" {i+1}", str(train_acc).ljust(20), str(test_acc).ljust(20), str(train_f1).ljust(20), str(test_f1).ljust(20))
    else:
        print(i+1, str(train_acc).ljust(20), str(test_acc).ljust(20), str(train_f1).ljust(20), str(test_f1).ljust(20))

print("\n\n-----------------------------------------------------------------------")
print("-------------------------------done------------------------------------")
print("-----------------------------------------------------------------------")