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
import ast
import torch

torch.set_float32_matmul_precision('high')

clean = False
bool_sampling = False
lang_list = ['python']
num_shots = 1 
seed = 42
model_name_dict = {
    # "starcoder": ["starcoderbase-7b"],
    # "octopack": ["octogeex"],
    "opencoder": ["OpenCoder-1.5B-Instruct", "OpenCoder-8B-Instruct"],
    "magicoder": ["Magicoder-CL-7B", "Magicoder-DS-6.7B"]
}
clone_type_list = ['original', 'Type I', 'Type II', 'Type III']
random.seed(seed)

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

def extract_variable_names(code: str) -> dict:
    try:
        tree = ast.parse(code)
        variable_info = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                var_name = node.id
                if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
                    lines = code.split('\n')
                    if 0 <= node.lineno - 1 < len(lines):
                        start_pos = node.col_offset
                        abs_start = sum(len(lines[i]) + 1 for i in range(node.lineno - 1) + start_pos)
                        abs_end = abs_start + len(var_name)
                        variable_info[var_name] = variable_info.get(var_name, []) + [(abs_start, abs_end)]
            elif isinstance(node, ast.FunctionDef):
                var_name = node.name
                if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
                    lines = code.split('\n')
                    if 0 <= node.lineno - 1 < len(lines):
                        line = lines[node.lineno - 1]
                        start_pos = line.find("def ", 0, node.col_offset) + 4
                        if start_pos >= 4:
                            abs_start = sum(len(lines[i]) + 1 for i in range(node.lineno - 1)) + start_pos
                            abs_end = abs_start + len(var_name)
                            variable_info[var_name] = variable_info.get(var_name, []) + [(abs_start, abs_end)]
            elif isinstance(node, ast.ClassDef):
                var_name = node.name
                if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
                    lines = code.split('\n')
                    if 0 <= node.lineno - 1 < len(lines):
                        line = lines[node.lineno - 1]
                        start_pos = line.find('class ', 0, node.col_offset) + 6
                        if start_pos >= 6:
                            abs_start = sum(len(lines[i]) + 1 for i in range(node.lineno - 1)) + start_pos
                            abs_end = abs_start + len(var_name)
                            variable_info[var_name] = variable_info.get(var_name, []) + [(abs_start, abs_end)]
            
        return variable_info
    except:
        return {}

def get_closing_token_mask(tokenizer, text: str, input_ids: torch.Tensor) -> torch.Tensor:
    DATA_MODEL_SYMBOLS = {"]", "}", ")", ".", "'", ","}
    EXPRESSIONS_KEYWORDS = {"in", "else", "is", "not"}
    EXPRESSIONS_SYSBOLS = {"]", "}", ")", ":"}
    SINGLE_STMT_KEYWORDS = {"as", "import", "from", "global", "nonlocal", "assert"}
    SINGLE_STMT_SYMBOLS = {","}
    COMPOUND_KEYWORDS = {"if", "for", "while", "try", "with", "class", "def", "elif", "else", "except", "finally", "match", "case", "async"}
    FUNC_ARG_TOKENS = {"self", "/", "*"}
    ARROW_TOKENS = {"->", "-", ">"}
    CLOSE_KEYWORDS = set().union(EXPRESSIONS_KEYWORDS, SINGLE_STMT_KEYWORDS, COMPOUND_KEYWORDS, FUNC_ARG_TOKENS)
    CLOSE_SYMBOLS = set().union(DATA_MODEL_SYMBOLS, EXPRESSIONS_SYSBOLS, SINGLE_STMT_SYMBOLS, ARROW_TOKENS, {":", ";", ","})
    mask = torch.ones(input_ids.shape[1] - 1, dtype = torch.bool)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0][1:])
    # encoding = tokenizer.encode_plus(
    #     text,
    #     # return_offsets_mapping=True,
    #     add_special_tokens=True,
    #     max_length=len(input_ids[0]),
    #     truncation=True,
    #     padding="max_length"
    # )
    # offset_mapping = encoding['offset_mapping'][1:len(tokens)+1]
    variable_positions = extract_variable_names(text)

    for i, token  in enumerate(tokens):
        if token == '[PAD]' or token == tokenizer.eos_token:
            mask[i] = False
            continue
        if token.strip() != token:
            mask[i] = False
            continue
        # if i < len(offset_mapping):
        #     token_start, token_end = offset_mapping[i]
        bare = token.strip()
        if bare in CLOSE_KEYWORDS or bare in CLOSE_SYMBOLS:
            mask[i] = False
        elif token.strip() == "_":
            mask[i] = False
        elif token.strip() == "." and i > 0 and i < len(tokens) - 1:
            prev_token = token[i - 1].strip()
            if prev_token == 'self' or any(prev_token == var_name for var_name in variable_positions):
                mask[i] = False

    return mask

def safe_compute_scores(token_log_probs: torch.Tensor, mask: torch.Tensor, ratio: float) -> float:
    valid_probs = token_log_probs[mask]
    if len(valid_probs) == 0:
        return 0.0
    
    valid_probs = valid_probs - valid_probs.max()
    valid_probs = torch.exp(valid_probs)
    valid_probs = valid_probs / valid_probs.sum()
    k = min(int(len(valid_probs) * ratio), len(valid_probs))
    return 0.0 if k == 0 else torch.mean(torch.sort(valid_probs)[0][:k]).item()

def eval(model, tokenizer, data, max_length):
    scores_dict = defaultdict(list)
    for code in tqdm(data, total = len(data)):
        if len(code) >= 60000: code = code[:60000]
        # if tokenizer.pad_token is None:
        #     tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        # else:
        #     tokenizer.pad_token = tokenizer.eos_token

        encoding = tokenizer.encode_plus(
            code, 
            max_length = max_length,
            truncation = True,
            return_tensors = 'pt',
            # return_offsets_mapping = True,
            # padding = 'max_length',
            # pad_to_multiple_of = 8
        )
        inp = encoding["input_ids"].to(model.device)
        attention_mask = encoding['attention_mask'].to(model.device)

        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
            outputs_mask = model(inp, labels = inp, output_hidden_states = True, attention_mask = attention_mask)
            logits_mask = outputs_mask.logits
            
            mask = get_closing_token_mask(tokenizer, code, inp)
            log_probs_mask = F.log_softmax(logits_mask[0, :-1], dim = -1)
            masked_log_probs = log_probs_mask.clone()
            masked_log_probs[~mask] = float('-inf')
            masked_log_probs = F.log_softmax(masked_log_probs, dim = -1)
            input_ids_for_mask = inp[0][1:].unsqueeze(-1)
            token_lp = masked_log_probs.gather(dim = -1, index = input_ids_for_mask).squeeze(-1)
        
        scores_dict['synprune'].append(safe_compute_scores(token_lp, mask, 0.1))

    return scores_dict

# test_code = """
# if __name__ == "__main__"
#     print("Hello world")
# """

def data_process(dataset_path: str, clone_type: str = "None", bool_sampling = False, sample_size: int = 100):
    dataset = read_jsonl(dataset_path)
    random.shuffle(dataset)

    data, labels = [], []
    for item in dataset:
        item_type = item['type']
        if item_type == 'Type_III': item_type = 'Type III'
        # if item['type'] == 'Type_III': item['type'] = 'Type III'
        
        if item_type != clone_type: continue
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

def calculate(scores, labels):
    results = defaultdict(list)
    for method, scores in scores.items():
        fpr, tpr, auc, acc = get_metrics(scores, labels)
        low = tpr[np.where(fpr<.01)[0][-1]]
        results['method'].append(method)
        results['auc'].append(f'{auc:.4%}')
        results['accuracy'].append(f'{acc:.4%}')
        results['tpr01'].append(f'{low:.4%}')

    df = DataFrame(results)   
    return df
