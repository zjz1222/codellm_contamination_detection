import json
from typing import Optional
import re

def read_jsonl(dataset_path: str):
    data = []
    with open(dataset_path, 'r', encoding = 'utf-8', errors = 'ignore') as file: 
        for idx, line in enumerate(file):
            data.append(json.loads(line.strip()))
    return data

def write_jsonl(json_list, output_file):
    # print(len(json_list))
    with open(output_file, 'w', encoding = 'utf-8') as f:
        for item in json_list:
            json.dump(item, f, ensure_ascii = False)
            f.write('\n')

def parse_code_block(string: str, lang: str = 'python') -> Optional[str]:
    pattern = re.compile(rf'```{lang}\s*\n(.*?)```', re.DOTALL | re.IGNORECASE)
    match = pattern.search(string)
    
    if match:
        code = match.group(1).strip()
        return code
    
    return None

def get_type_feature(lang: str):
    type_features = []
    if lang == 'cpp':
        type_features = ['whitespace', 'newline', 'identifier', ';', ')', '(', 'number_literal', 'primitive_type', '}', '{', '=', 'field_identifier', 'type_identifier', ',', '[', ']', '.', '<', '"', '<<', '::', 'namespace_identifier', '>', 'if', '+', 'for', 'comment', "'", 'return', '>>', '-', 'string_content', ':', '++', '*', '#include', '&', 'system_lib_string', '==', 'long', 'const', 'escape_sequence', 'false', 'while', 'else', 'character', '--', '<=', 'auto', 'using', 'namespace', '%', '!=', '&&', 'nullptr', '->', '+=', 'true', '/', '!', '>=', 'continue', 'class', 'public', 'break', '?', '||', 'private', 'struct', 'new', '-=', 'typename', 'template', 'this', 'static', 'case', 'operator', 'override', 'delete', 'virtual', '~', 'unsigned', 'throw', '/=', '^', '|', 'sizeof', 'enum', 'typedef', '>>=', '...', 'catch', '#define', '|=', '*=', 'try', 'NULL', 'preproc_arg', 'default', 'switch', '%=', 'statement_identifier', 'constexpr', 'noexcept', '#endif', 'goto', '()', '#ifndef', 'L"', 'explicit', '^=', 'do', 'short', 'R"', 'raw_string_content', 'protected', '<<=', '&=', 'inline', 'literal_suffix', 'decltype', '[]', 'preproc_directive', '[[', ']]', 'and', '#ifdef', 'static_assert', 'friend', '#else', 'extern', "L'", '\n', 'alignof', 'mutable', '#if', 'offsetof', 'raw_string_delimiter', 'volatile', 'restrict', 'defined', 'not', 'signed', '#elif']
    if lang == 'python':
        type_features = ['whitespace', 'newline', 'class', 'identifier', ':', 'def', '(', ')', 'pass', ',', '*', '**', 'if', '<', 'integer', 'string_start', 'string_content', 'string_end', 'else', '=', '[', ']', '==', '{', '}', 'elif', '.', 'comment', 'import', 'as', '+', 'for', 'in', '->', 'return', 'from', 'ellipsis', 'try', 'and', 'except', 'none', 'false', 'true', '@', 'assert', '+=', '-', '>=', '-=', 'not', 'or', '/', '\\', 'raise', 'with', 'float', 'del', '>', 'while', 'break', 'block', 'lambda', 'is', '%', '!=', 'continue', 'escape_interpolation', 'escape_sequence', 'async', 'await', '<=', 'global', '&', '&=', '|=', '*=', '//=', '/=', '<<', ';', '//', '|', 'line_continuation', '<>', 'yield', 'finally', 'print', '~', '^', 'nonlocal', '<<=', 'type_conversion', '>>', '^=']
    print(len(type_features))
    return type_features

def get_max_length(model_name: str):
    max_length = 8192
    if model_name == "OpenCoder-1.5B-Instruct":
        max_length = 4096
    if model_name == 'starcoderbase' or model_name == 'octocoder':
        max_length = 4096
    return max_length
sample_text = """
Python example:

```python
def hello_world():
    print("Hello, World!")
    return True
```

JavaScript example:

```javascript
function hello() {
    console.log("Hello");
}
```

Another Python example:

```python
def calculate_sum(a, b):
    return a + b
```
"""


if __name__ == "__main__":
    # print(parse_code_block(sample_text, lang = 'python'))
    print(get_type_feature('cpp'))
    print(get_type_feature('python'))