# Release path adapter; original experiment functions are retained.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1]))
from _release_paths import release_path as _release_path

import os
# Set CUDA_VISIBLE_DEVICES in the shell for your machine.

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast
import sys
import torch
import torch.nn.functional as F
import difflib
from utils import read_jsonl, get_type_feature, get_max_length
import re
import gc

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import read_jsonl
from code_parser import code_parse
from collections import defaultdict
import numpy as np
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm
import math
import random
from torch.amp import autocast
from tokenizers import Tokenizer


# Basic settings.
model_name_dict = {
    # "starcoder": ["starcoderbase", "starcoderbase-7b"],
    "magicoder": ['Magicoder-DS-6.7B'],
    # "octopack": ["octocoder", "octogeex"],
    # "opencoder": ["OpenCoder-1.5B-Instruct", "OpenCoder-8B-Instruct"],
}
lang_list = ['python']
min_k_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
configures = ['combine']

# model_name_dict = {
#     "magicoder": ['Magicoder-CL-7B', 'Magicoder-DS-6.7B'],
#     "octopack": ["octocoder", "octogeex"],
#     "opencoder": ["OpenCoder-1.5B-Instruct", "OpenCoder-8B-Instruct"],
#     "starcoder": ["starcoderbase", "starcoderbase-7b"]
# }
# lang_list = ['python', 'cpp']
# min_k = 0.3
# configures = ['normalized_log_prob', 'log_prob', 'combine'] 
SPECIAL_VALUE = 998244353

def get_transformers_model(model_path: str):
    device_map = "auto"
    if "octogeex" in model_path: 
        device_map = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, trust_remote_code=True)
    model.eval()
    return model, tokenizer

def token_probability_parse(model, tokenizer, text, max_length,eps = 1e-6):
    torch.cuda.empty_cache()
    if tokenizer.is_fast:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=False, return_offsets_mapping = True).to(model.device)
        offset_mapping = inputs['offset_mapping'][0].cpu().numpy()
        inputs.pop("offset_mapping")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=False).to(model.device)
    
    nofast_prefix = "" 
    
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(**inputs, labels = inputs["input_ids"].long())
        input_ids = inputs["input_ids"].long()
        
        results = []

        first_token_id = input_ids[0, 0].item()
        first_token = tokenizer.decode(first_token_id)
        # first_start, first_end = offset_mapping[0]
        # first_prefix = text[:first_end]

        results.append({
            "input_id": first_token_id,
            "token": first_token,
            "token_log_prob": SPECIAL_VALUE,
            "prefix": first_token
        })
        nofast_prefix = first_token

        loss = outputs.loss
        logits = outputs.logits

        seq_logits = outputs.logits[0, :-1]  
        target_ids = input_ids[0, 1:].unsqueeze(-1)  

        token_log_probs = -F.cross_entropy(
            seq_logits, 
            target_ids.squeeze(-1), 
            reduction="none"
        )
        log_probs = F.log_softmax(seq_logits, dim=-1)
        probs = F.softmax(seq_logits, dim=-1)
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
        mink_plus = (token_log_probs - mu) / (sigma.sqrt() + eps)

        del seq_logits, probs, log_probs, mu, sigma, outputs
        torch.cuda.empty_cache()


    input_ids_list = target_ids.squeeze(-1).tolist()
    tokens = [tokenizer.decode([id_]) for id_ in input_ids_list]
    token_log_probs = token_log_probs.tolist()
    mink_plus = mink_plus.tolist()
    
    if tokenizer.is_fast:
        for idx in range(len(input_ids_list)):
            current_offset = offset_mapping[1 + idx]
            current_end = current_offset[1]
            current_prefix = text[:current_end]

            results.append({
                'input_id': input_ids_list[idx],
                'token': tokens[idx],
                'token_log_prob': token_log_probs[idx],
                'normalized_log_prob': mink_plus[idx],
                'prefix': current_prefix
            })
    else:
        for idx in range(len(input_ids_list)):
            if tokens[idx] == '<0x0A>': tokens[idx] = '\n'

            nofast_prefix += tokens[idx]
            results.append({
                'input_id': input_ids_list[idx],
                'token': tokens[idx],
                'token_log_prob': token_log_probs[idx],
                'normalized_log_prob': mink_plus[idx],
                'prefix': nofast_prefix
            })
        
    return results

def longest_common_substring(str1, str2):
    seq_match = difflib.SequenceMatcher(None, str1, str2)
    match = seq_match.find_longest_match(0, len(str1), 0, len(str2))
    
    if match.size > 0:
        return match.size / len(str2)
    return 0

from difflib import SequenceMatcher

def get_lcs_length(str1, str2):
    s = SequenceMatcher(None, str1, str2)
    match = s.find_longest_match(0, len(str1), 0, len(str2))
    return match.size

def bind(token_probability_results, code_parse_results):
    # print(token_probability_results)
    # text_tokens = [item['token'] for item in token_probability_results]
    # print(text_tokens)
    # code_tokens = [item['token'] for item in code_parse_results]
    # print(code_tokens)
    # code_types = [item['type'] for item in code_parse_results]
    # print(code_types)

    results = []
    pointer = 0
    text_record = ""
    for idx in range(len(token_probability_results)):
        item = token_probability_results[idx]
        text_record = item['prefix']
        text_token = item['token']
            
        max_lcs, token_type = get_lcs_length(text_token, code_parse_results[pointer]['token']), code_parse_results[pointer]['type']

        while (len(code_parse_results[pointer]['prefix']) + len(code_parse_results[pointer]['token']) <= len(text_record))\
              and pointer < len(code_parse_results) - 1:
            pointer = pointer + 1 
            lcs = get_lcs_length(text_token, code_parse_results[pointer]['token'])
            if lcs > max_lcs:
                token_type = code_parse_results[pointer]['type']
                max_lcs = lcs

        if (text_token.isspace() or text_token == '') and '\n' not in text_token: 
            item['type'] = 'whitespace'
            results.append(item)
            continue
        if text_token.isspace() and ('\n' in text_token or '\r' in text_token):
            item['type'] = 'newline'
            results.append(item)
            continue


        item['type'] = token_type
        results.append(item)

    return results

def get_code_features(model, tokenizer, text, type_features, max_length, configure, min_k, eps = 1e-6):   
    if len(text) >= 60000: text = text[:60000]

    code_parse_results = code_parse(text)
    token_probability_results = token_probability_parse(model, tokenizer, text, max_length, eps)   

    if token_probability_results != None and token_probability_results != []:
        tokens = bind(token_probability_results, code_parse_results)
    else:
        return None
    # for token in tokens:
    #     print(f"{token['token']} | {token['type']}")
    features = []

    type_dicts = defaultdict(list)
    for token in tokens:
        token_type = token['type']
        token_log_prob = token['token_log_prob']

        if token_log_prob == SPECIAL_VALUE: continue
        if configure == 'log_prob' or configure == 'combine':
            type_dicts[f"{token_type}_log_prob"].append(token_log_prob)

        if configure == 'normalized_log_prob' or configure == 'combine':
            normalized_token_log_prob = token['normalized_log_prob']
            type_dicts[f"{token_type}_normalized_log_prob"].append(normalized_token_log_prob)

    for type_feature in type_features:
        if configure == 'log_prob' or configure == 'combine':
            type_values = type_dicts.get(f"{type_feature}_log_prob", [])
            if type_values == []:
                features.append(np.nan)
            elif min_k == 1.0:
                features.append(sum(type_values) / len(type_values))
            else:
                token_log_probs = np.array(type_values)
                k_length = math.ceil(len(token_log_probs) * min_k)
                topk = np.sort(token_log_probs)[:k_length]
                features.append(np.mean(topk).item())

                # if type_feature == "identifier":
                #     print(f"identifier logp.: {np.mean(topk).item()}")
                # if type_feature == "while":
                #     print(f"while logp.: {np.mean(topk).item()}")
                # if type_feature == "<=":
                #     print(f"<= logp.: {np.mean(topk).item()}")

        if configure == 'normalized_log_prob' or configure == 'combine':
            type_values = type_dicts.get(f"{type_feature}_normalized_log_prob", [])
            if type_values == []:
                features.append(np.nan)
            elif min_k == 1.0:
                features.append(sum(type_values) / len(type_values))
            else:
                token_log_probs = np.array(type_values)
                k_length = math.ceil(len(token_log_probs) * min_k)
                topk = np.sort(token_log_probs)[:k_length]
                features.append(np.mean(topk).item())
                
                # if type_feature == "identifier":
                #     print(f"identifier norm logp.: {np.mean(topk).item()}")
                # if type_feature == "while":
                #     print(f"while norm logp.: {np.mean(topk).item()}")
                # if type_feature == "<=":
                #     print(f"<= norm logp.: {np.mean(topk).item()}")

    return features

    
async def dataset_process(model_path, input_path, output_path, type_features, max_length, configure, min_k):
    model, tokenizer = get_transformers_model(model_path)
    data = read_jsonl(input_path)
    
    random.seed(42)  
    feature_names = []
    for type_feature in type_features:
        if configure == "log_prob" or configure == "combine":
            feature_names.append(f"{type_feature}_log_prob")
        if configure == "normalized_log_prob" or configure == "combine":
            feature_names.append(f"{type_feature}_normalized_log_prob")
    
    write_features, write_labels, write_clone_types = [], [], []
    for item in tqdm(data, desc="Feature extraction:"):
        text, label, clone_type = item['code'], item['label'], item['type']
        if text.startswith("Error"): continue
        features = get_code_features(model, tokenizer, text, type_features, max_length, configure, min_k)

        if features is None: continue
        write_features.append(features)
        write_labels.append(label)
        write_clone_types.append(clone_type)
    
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() 

    df = pd.DataFrame(write_features, columns = feature_names)
    df['target'] = write_labels
    df['clone_type'] = write_clone_types

    df.to_parquet(output_path)
