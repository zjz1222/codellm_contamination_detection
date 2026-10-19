# Release path adapter; original experiment functions are retained.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1]))
from _release_paths import release_path as _release_path

import os
# Set CUDA_VISIBLE_DEVICES in the shell for your machine.

import sys
from collections import defaultdict
from tqdm import tqdm
import torch
import zlib
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, auc
from pandas import DataFrame
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from utils import read_jsonl
import random
from torch.amp import autocast
import gc
import math
import pandas as pd

torch.set_float32_matmul_precision('high')

clean = True
bool_sampling = False
lang_list = ['python']
num_shots = 1 
seed = 42
model_name_dict = {
    "starcoder": ["starcoderbase"],
    "magicoder": ["Magicoder-DS-6.7B"],
    "octopack": ["octocoder"],
    "opencoder": ["OpenCoder-8B-Instruct"],
}
clone_type_list = ['original', 'Type I', 'Type II', 'Type III']

def get_max_length(model_name: str):
    max_length = 8192
    if model_name == "OpenCoder-1.5B-Instruct":
        max_length = 4096
    if model_name == 'starcoderbase' or model_name == 'octocoder':
        max_length = 4096
    return max_length

def get_metrics(scores, labels):
    def safe_isnan(val):
        try:
            return math.isnan(float(val))
        except (ValueError, TypeError):
            return True 

    filtered_data = [(s, l) for s, l in zip(scores, labels) if not safe_isnan(s)]
    scores, labels = zip(*filtered_data)
    scores = [float(s) for s in scores] 

    fpr, tpr, _ = roc_curve(labels, scores)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc

def get_transformers_model(model_path: str):
    device_map = "auto"
    if "octogeex" in model_path: device_map = "cuda:0"

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    return model, tokenizer
    
def fix_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

def get_all_prob(input_ids, loss, logits):
    probabilities = torch.nn.functional.log_softmax(logits, dim = -1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)

    ll = -loss.item()
    ppl = torch.exp(loss).item()
    prob = torch.exp(-loss).item()
    return prob, ll, ppl, all_prob, loss.item()

def get_ll(sentence, model, tokenizer, max_length):
    input_encodings = tokenizer(sentence, return_tensors = "pt", max_length = max_length, truncation=True, add_special_tokens = False).to(model.device)
    input_ids = input_encodings.input_ids.long()

    with torch.no_grad():
        outputs = model(input_ids, labels = input_ids)
    loss, logits = outputs.loss, outputs.logits
    return get_all_prob(input_ids, loss, logits)

def get_conditional_ll(input_text, target_text, model, tokenizer, max_length):
    input_encodings = tokenizer(input_text, return_tensors = "pt", max_length = max_length, truncation=True, add_special_tokens = False).to(model.device)
    target_encodings = tokenizer(target_text, return_tensors = "pt", max_length = max_length, truncation=True, add_special_tokens = False).to(model.device)
    concat_ids = torch.cat((input_encodings.input_ids.long(), target_encodings.input_ids.long()), dim = 1)[:, :max_length]
    labels = concat_ids.clone()
    labels[:, : input_encodings.input_ids.size(1)] = -100
    with torch.no_grad():
        outputs = model(concat_ids, labels = labels)
    
    loss, logits = outputs.loss, outputs.logits
    return get_all_prob(labels, loss, logits)

def process_prefix(tokenizer, prefix, avg_length, pass_window, total_shots, max_length):
    if pass_window:
        return prefix
    token_counts = [len(tokenizer.encode(shot)) for shot in prefix]
    target_token_count = avg_length

    total_tokens = sum(token_counts) + target_token_count
    if total_tokens <= max_length:
        return prefix

    max_shots = 0
    cumulative_tokens = target_token_count
    for count in token_counts:
        if cumulative_tokens + count < max_length:
            max_shots += 1
            cumulative_tokens += count

        else:
            break
    
    truncated_prefix = prefix[:max_shots]
    total_shots = max_shots
    return truncated_prefix

def predict(model, tokenizer, data, max_length, eps = 1e-6, prefix: str = ["Hello Word"]):
    scores = defaultdict(list)
    for index, text in enumerate(tqdm(data, total = len(data), desc = 'Samples')):
        if len(text) >= 60000: text = text[:60000]

        inputs = tokenizer(text, return_tensors = "pt", max_length = max_length, truncation=True, add_special_tokens = False).to(model.device)
        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(**inputs, labels = inputs["input_ids"].long())
            input_ids = inputs["input_ids"].long()
            
            loss = outputs.loss
            logits = outputs.logits
            
            ll = -loss.item()
            scores['loss'].append(ll)
            scores['zlib'].append(ll / (len(zlib.compress(bytes(text, 'utf-8'))) + eps))
                    
            input_ids = input_ids[0][1:].unsqueeze(-1)
            probs = F.softmax(logits[0, :-1], dim = -1)
            log_probs = F.log_softmax(logits[0,:-1], dim = -1)
            token_log_probs = log_probs.gather(dim = -1, index = input_ids).squeeze(-1)

            mu = (probs * log_probs).sum(-1)
            sigma = ((probs * torch.square(log_probs)).sum(-1) - torch.square(mu))

            for ratio in [0.1, 0.2, 0.3]:
                k_length = int(len(token_log_probs) * ratio)
                topk = np.sort(token_log_probs.cpu().detach().numpy())[:k_length]
                if k_length == 0:
                    scores[f'mink_{ratio}'].append(0.0)
                else:
                    scores[f'mink_{ratio}'].append(np.mean(topk).item())
                
            
            mink_plus = (token_log_probs - mu) / (sigma.sqrt() + eps)

            for ratio in [0.1, 0.2, 0.3]:
                k_length = int(len(mink_plus) * ratio)
                topk = np.sort(mink_plus.cpu().detach().numpy())[:k_length]
                if k_length == 0:
                    scores[f'mink_{ratio}'].append(0.0)
                else:
                    scores[f'mink++_{ratio}'].append(np.mean(topk).item())

            ll_nonmember = get_conditional_ll("\n".join(prefix), text, model, tokenizer, max_length)[1]
            scores['recall'].append(ll_nonmember / (ll + eps))

    return scores
        
def calculate(scores, labels):
    results = defaultdict(list)
    for method, scores in scores.items():
        fpr, tpr, auc, acc = get_metrics(scores, labels)
        low = tpr[np.where(fpr<.01)[0][-1]]
        results['method'].append(method)
        results['auc'].append(auc)
        results['accuracy'].append(acc)
        results['tpr01'].append(low)

    df = DataFrame(results)   
    return df 


def data_process(dataset_path: str,  bool_sampling = False, sample_size: int = 100, clone_type=None ):
    dataset = read_jsonl(dataset_path)
    random.shuffle(dataset)

    data, labels = [], []
    for item in dataset:
        # item_type = item['type']
        # if item_type == 'Type_III': item_type = 'Type III'
        
        # if item_type != clone_type: continue
        code = item['code']
        if code.startswith("Error"): continue

        data.append(code)
        labels.append(item['label'])
    
    if bool_sampling == False or len(data) <= sample_size: return data, labels

    indices = list(range(len(data)))
    sampled_indices = random.sample(indices, sample_size)
    data_sample = [data[i] for i in sampled_indices]
    labels_sample = [labels[i] for i in sampled_indices]

    return data_sample, labels_sample

def get_prefix(tokenizer,dataset_path, max_length):
    data, labels = data_process(dataset_path, bool_sampling)
    prefix = [data[idx] for idx in range(len(data)) if labels[idx] == 0]
    prefix = random.sample(prefix, k = num_shots)
    avg_length = int(np.mean([len(tokenizer.encode(item)) for item in data]))
    prefix = process_prefix(tokenizer, prefix, avg_length, pass_window = False, total_shots = num_shots, max_length = max_length)

    return prefix

def file_write_brief(output_path: str, clone_type: str, results: DataFrame):
    results_print = results.copy()
    fmt = lambda x: f'{x*100:.1f}'
    print(results_print.style.format({'auc': fmt, 'accuracy': fmt, 'tpr01': fmt}).to_string())
    with open(output_path, 'a', encoding = 'utf-8') as file:
        file.write(f"======== {clone_type} ========\n")
        df_copy = results.copy()
        df_copy['auc'] = df_copy['auc'].apply(lambda x: f'{x*100:.1f}')
        df_copy['accuracy'] = df_copy['accuracy'].apply(lambda x: f'{x*100:.1f}')
        df_copy['tpr01'] = df_copy['tpr01'].apply(lambda x: f'{x*100:.1f}')

        file.write(df_copy.to_csv(index=False, sep=',', header=True))
        file.write("\n")

def file_write_full(output_path: str, results: DataFrame, clone_type = None):
    results_print = results.copy()
    print(results_print.style.format({'auc': '{:.4%}', 'accuracy': '{:.4%}', 'tpr01': '{:.4%}'}).to_string())
    with open(output_path, 'a', encoding = 'utf-8') as file:
        df_copy = results.copy()
        df_copy['auc'] = df_copy['auc'].apply(lambda x: f'{x:.4%}')
        df_copy['accuracy'] = df_copy['accuracy'].apply(lambda x: f'{x:.4%}')
        df_copy['tpr01'] = df_copy['tpr01'].apply(lambda x: f'{x:.4%}')

        file.write(df_copy.to_csv(index=False, sep=',', header=True))
        file.write("\n")

def file_write(output_brief_path: str, output_full_path: str, clone_type: str, results: DataFrame):
    file_write_brief(output_brief_path, clone_type, results)
    file_write_full(output_full_path, clone_type, results)
