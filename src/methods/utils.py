import json
from typing import Optional
import re

def read_jsonl(dataset_path: str):
    data = []
    with open(dataset_path, 'r', encoding = 'utf-8', errors = 'ignore') as file: 
        for line in file:
            data.append(json.loads(line.strip()))
    return data