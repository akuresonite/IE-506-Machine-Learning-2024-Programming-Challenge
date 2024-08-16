import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz
import sys
from threading import Thread
import os
import joblib
from joblib import parallel_config, Parallel, delayed
from sklearn.preprocessing import StandardScaler
from scipy.sparse import dok_matrix



### Extracting Classes \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
def process_labels(file_path):
    M, S = set(), set()
    data = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Extracting labels", colour='GREEN'):
            main_label_part, sub_label_part = line.strip().split(' ')[:2]
            m = set(int(i) for i in main_label_part.split(":")[1].split(","))
            s = set(int(i) for i in sub_label_part.split(":")[1].split(","))
            M.update(m)
            S.update(s)
            data.append((m, s))

    cols_M = [f"M_{i}" for i in sorted(M)]
    cols_S = [f"S_{i}" for i in sorted(S)]
    cols_C = cols_M + cols_S

    label_array = dok_matrix((len(data), len(cols_C)), dtype=int)
    
    for i, (m, s) in enumerate(tqdm(data, desc="Filling label values", colour='RED')):
        for v2 in cols_M:
            if int(v2.split("_")[1]) in m:
                label_array[i, cols_C.index(v2)] = 1

        for v2 in cols_S:
            if int(v2.split("_")[1]) in s:
                label_array[i, cols_C.index(v2)] = 1

    return label_array, cols_C


### Extracting features \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
def process_features(file_path):
    features = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Extracting features", colour='GREEN'):
            feature_parts = line.strip().split(' ')[2:]
            feature_dict = {int(f.split(':')[0]): float(f.split(':')[1]) for f in feature_parts}
            features.append(feature_dict)

    max_feature_index = max(max(d.keys()) for d in features)
    cols_F = [f'F_{i}' for i in range(max_feature_index + 1)]
    feature_array = dok_matrix((len(features), max_feature_index + 1), dtype=float)
    
    for i, d in enumerate(tqdm(features, desc="Filling features values", colour='RED')):
        for feature, value in d.items():
            feature_array[i, feature] = value
    
    return feature_array, cols_F



if __name__ == '__main__':
    
    dataset_path = r"/home/23m1521/ashish/Kaggle/_3_IE506_2024_Programming_Challenge/dataset"
    train_txt_path  = r"/home/23m1521/ashish/Kaggle/_3_IE506_2024_Programming_Challenge/dataset/IE506_2024_progchallenge_train.txt"
    test_txt_path = r"/home/23m1521/ashish/Kaggle/_3_IE506_2024_Programming_Challenge/dataset/IE506_2024_progchallenge_test.txt"
    sample_txt_path = r"/home/23m1521/ashish/Kaggle/_3_IE506_2024_Programming_Challenge/dataset/sample.txt" 
    sample2_txt_path = r"/home/23m1521/ashish/Kaggle/_3_IE506_2024_Programming_Challenge/dataset/sample2.txt" 
    sample_test_txt_path = r"/home/23m1521/ashish/Kaggle/_3_IE506_2024_Programming_Challenge/dataset/sample_test.txt"
    
    ### Making Train-Test DataFrame \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    file_path = train_txt_path
    
    csr_label, cols_C = process_labels(file_path)
    print("Labels DataFrame created.\n")

    csr_feature, cols_F = process_features(file_path)
    print("Features DataFrame created.\n")
    
    csr_feature_sub, cols_F_sub = process_features(test_txt_path)
    print("Test Features DataFrame created.\n")
        
        
    # Print the memory size in MB of csr_matrix(label_array), label_array, csr_matrix(feature_array), feature_array
    print(f"Memory size of csr_matrix(label_array) in MB: {sys.getsizeof(csr_label) / (1024 * 1024)}")
    print(f"Memory size of csr_matrix(feature_array) in MB: {sys.getsizeof(csr_feature) / (1024 * 1024)}")
    print(f"Memory size of csr_matrix(feature_array) in MB: {sys.getsizeof(csr_feature_sub.data) / (1024 * 1024)}")
    
    csr_label = csr_matrix(csr_label)
    csr_feature = csr_matrix(csr_feature)
    
    print(f"Memory size of csr_matrix(label_array) in MB: {sys.getsizeof(csr_label) / (1024 * 1024)}")
    print(f"Memory size of csr_matrix(feature_array) in MB: {sys.getsizeof(csr_feature) / (1024 * 1024)}")
    print(f"Memory size of csr_matrix(feature_array) in MB: {sys.getsizeof(csr_feature_sub.data) / (1024 * 1024)}")

    
    # Save the CSR matrices to disk
    print("------------npz-------------------")
    save_npz(f"{dataset_path}/csr_label.npz", csr_label)
    save_npz(f"{dataset_path}/csr_feature.npz", csr_feature)
    save_npz(f"{dataset_path}/csr_feature_sub.npz", csr_feature_sub)
    
    # Print saved csr matrices
    print(f"\nSaved csr_label: {os.path.getsize(f'{dataset_path}/csr_label.npz') / (1024**2)} MB")
    print(f"Saved csr_feature: {os.path.getsize(f'{dataset_path}/csr_feature.npz') / (1024**2)} MB")
    print(f"Saved csr_feature_sub: {os.path.getsize(f'{dataset_path}/csr_feature_sub.joblib') / (1024**2)} MB")
    
    # Save cols_C and cols_F to disk
    joblib.dump(cols_C, f"{dataset_path}/cols_C.joblib")
    joblib.dump(cols_F, f"{dataset_path}/cols_F.joblib")
    joblib.dump(cols_F_sub, f"{dataset_path}/cols_F_sub.joblib")
