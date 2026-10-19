# Release path adapter; original experiment functions are retained.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1]))
from _release_paths import release_path as _release_path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
from utils import get_type_feature
import math
from pandas import DataFrame


train_dataset_size = 1000
configures = ["combine", "log_prob", "normalized_log_prob"]
seed = 42
min_k = 0.1
model_name_dict = {
    "magicoder": {"Magicoder-CL-7B", "Magicoder-DS-6.7B"},
    "octopack": {"octocoder", "octogeex"},
    "opencoder": {"OpenCoder-1.5B-Instruct", "OpenCoder-8B-Instruct"},
    "starcoder": {"starcoderbase", "starcoderbase-7b"}
}
lang_list = ['python', 'cpp']

# train_dataset_size = 1000
# configures = ["combine"]
# seed = 42
# min_k = 0.5
# model_name_dict = {
#      "magicoder": {"Magicoder-CL-7B"},
#  }
# lang_list = ['python']

# LightGBM Params
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',

    'num_leaves': 50,
    'max_depth': -1,
    'min_data_in_leaf': 15,
    
    'learning_rate': 0.1,
    'feature_fraction': 1.0,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,

    'lambda_l1': 0.2,
    'lambda_l2': 5.0,

    'path_smooth': 2.0,
    'extra_trees': True,

    'use_missing': True,
    'zero_as_missing': False,
    'verbose': -1,
    'seed': 42
}

def get_metrics(scores, labels):
    def safe_isnan(val):
        try:
            return math.isnan(float(val))
        except (ValueError, TypeError):
            return True 
    # print(scores)
    # print(labels)

    filtered_data = [(s, l) for s, l in zip(scores, labels) if not safe_isnan(s)]
    scores, labels = zip(*filtered_data)
    scores = [float(s) for s in scores] 

    fpr, tpr, _ = roc_curve(labels, scores)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc


def get_model(train_pkl_path, val_pkl_path, model_path, configure, type_features):
    df_train, df_test = pd.read_parquet(train_pkl_path), pd.read_parquet(val_pkl_path)
    y_train, y_test = df_train.pop("target"), df_test.pop("target")
    clone_type_train, clone_type_test = df_train.pop("clone_type"), df_test.pop("clone_type")

    del_suffix = ""
    if configure == 'normalized_log_prob': del_suffix = "log_prob"
    if configure == 'log_prob': del_suffix = "normalized_log_prob"
        
    for type_feature in type_features:
        col_name = f"{type_feature}_{del_suffix}"
        if col_name in df_train.columns: df_train.pop(col_name)
        if col_name in df_test.columns: df_test.pop(col_name)
     

    X_train, X_test = df_train, df_test
    
    old_features = X_train.columns
    new_features = [f'feature_{i}' for i in range(X_train.shape[1])]
    X_train.columns, X_test.columns = new_features, new_features
    
    # ---------- Training Sampling Begin --------------
    sample_num = min(train_dataset_size, len(X_train))  
    X_train_sub = X_train.sample(n=sample_num, random_state=42)  
    y_train_sub = y_train.loc[X_train_sub.index]  
    # old_features = X_train_sub.columns
    # new_features = [f'feature_{i}' for i in range(X_train_sub.shape[1])]
    # X_train_sub.columns, X_test.columns = new_features, new_features
    train_data = lgb.Dataset(X_train_sub, label=y_train_sub)
    # ---------- Training Sampling End --------------


    # train_data = lgb.Dataset(X_train, label = y_train)
    test_data = lgb.Dataset(X_test, label = y_test, reference = train_data)
    
    if not os.path.exists(model_path):
        model = lgb.train(
            params = params,
            train_set = train_data,
            num_boost_round = 500,
            valid_sets = [train_data, test_data],
            valid_names = ['train_set', 'valid_set'],
            callbacks = [
                lgb.early_stopping(stopping_rounds = 50),
                lgb.log_evaluation(period = 10)
            ]
        )
        model.save_model(model_path)
    else:
        model = lgb.Booster(model_file=model_path)
    
    return model, X_test, y_test, old_features, clone_type_test


def file_write_brief(output_path: str, results: DataFrame):
    results_print = results.copy()
    fmt = lambda x: f'{x*100:.1f}'
    print(results_print.style.format({'auc': fmt, 'accuracy': fmt, 'tpr01': fmt}).to_string())
    with open(output_path, 'a', encoding = 'utf-8') as file:
        df_copy = results.copy()
        df_copy['auc'] = df_copy['auc'].apply(lambda x: f'{x*100:.1f}')
        df_copy['accuracy'] = df_copy['accuracy'].apply(lambda x: f'{x*100:.1f}')
        df_copy['tpr01'] = df_copy['tpr01'].apply(lambda x: f'{x*100:.1f}')

        file.write(df_copy.to_csv(index=False, sep=',', header=True))
        file.write("\n")

def file_write_full(output_path: str, results: DataFrame):
    results_print = results.copy()
    print(results_print.style.format({'auc': '{:.4%}', 'accuracy': '{:.4%}', 'tpr01': '{:.4%}'}).to_string())
    with open(output_path, 'a', encoding = 'utf-8') as file:
        df_copy = results.copy()
        df_copy['auc'] = df_copy['auc'].apply(lambda x: f'{x:.4%}')
        df_copy['accuracy'] = df_copy['accuracy'].apply(lambda x: f'{x:.4%}')
        df_copy['tpr01'] = df_copy['tpr01'].apply(lambda x: f'{x:.4%}')

        file.write(df_copy.to_csv(index=False, sep=',', header=True))
        file.write("\n")

def file_write(output_brief_path: str, output_full_path: str,  results: DataFrame):
    file_write_brief(output_brief_path, results)
    file_write_full(output_full_path, results)

def test_lightgbm(model_series, model_name, lang, configure, type_features):
    print(f"Processing Model: {model_name}")
    
    os.makedirs(_release_path(f"data/{model_series}/lightgbm_models/{lang}"), exist_ok = True)

    train_pkl_path = _release_path(f"data/{model_series}/input_processed/{lang}/{model_name}_top{min_k}_train_data_clone.pkl")
    val_pkl_path = _release_path(f"data/{model_series}/input_processed/{lang}/{model_name}_top{min_k}_val_data_clone.pkl")
    model_path = _release_path(f"data/{model_series}/lightgbm_models/{lang}/{model_name}_{configure}_top{min_k}.txt")
    # output_path = f"/path/to/project"
    output_brief_path = _release_path(f"data/{model_series}/results/{lang}/{model_name}_top{min_k}_lightgbm_brief_result.txt")
    output_full_path = _release_path(f"data/{model_series}/results/{lang}/{model_name}_top{min_k}_lightgbm_full_result.txt")


    if not os.path.exists(train_pkl_path) or not os.path.exists(val_pkl_path): return

    model, X_test, y_test, old_features, y_clone_type = get_model(train_pkl_path, val_pkl_path, model_path, configure, type_features)    
    y_pred_proba = model.predict(X_test)

    columns = ["Clone Type", "auc", "accuracy", "tpr01"]
    clone_types = ['original', 'Type I', 'Type II', 'III']

    data = []
    mean_auc, mean_acc, mean_low = 0, 0, 0
    for clone_type in clone_types:
        y_pred_proba_part, y_test_part = [], []
        for idx in range(len(y_test)):
            if clone_type not in y_clone_type[idx]: continue
            
            y_pred_proba_part.append(y_pred_proba[idx])
            y_test_part.append(y_test[idx])
        
        fpr, tpr, auc, acc = get_metrics(y_pred_proba_part, y_test_part)
        low = tpr[np.where(fpr < .01)[0][-1]]
        data.append([clone_type, auc, acc, low])

        mean_auc += auc
        mean_acc += acc
        mean_low += low
        
    mean_auc, mean_acc, mean_low = mean_auc/4.0, mean_acc/4.0, mean_low/4.0
    data.append(['Macro Average', mean_auc, mean_acc, mean_low])

    results = pd.DataFrame(data = data, columns = columns)

    file_write(output_brief_path, output_full_path, results)
    # print(results)
    
    # with open(output_path, 'a', encoding = 'utf-8') as file:
    #     file.write(f"=========={configure}==========\n")
    #     file.write(results.to_csv(index=False, sep=',', header=True))
    #     file.write("\n")
    
    # file.write("Feature importance (top 20):\n")
    # head_num = 50
    # importance = pd.DataFrame({'feature': old_features, 'gain': model.feature_importance('gain')})
    # sorted_importance = importance.sort_values('gain', ascending=False).head(head_num)
    # file.write(sorted_importance.to_string())
    # file.write("\n\n")

if __name__ == "__main__":
    for model_series, model_name_list in model_name_dict.items():
        for model_name in model_name_list:
            for lang in lang_list:
                for configure in configures:
                    type_features = get_type_feature(lang)
                    test_lightgbm(model_series, model_name, lang, configure, type_features)