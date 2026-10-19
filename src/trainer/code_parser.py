from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp
import tree_sitter_python as tspython

"""
id
token:
predict_token:
type:
child_ids:
"""

def get_root_node(code: str, lang: str = 'python'):
    parser = None
    if lang == 'cpp':
        CPP_LANGUAGE = Language(tscpp.language())
        parser = Parser(CPP_LANGUAGE)
    
    if lang == 'python':
        PY_LANGUAGE = Language(tspython.language())
        parser = Parser(PY_LANGUAGE)
    
    if parser is None:
        raise ValueError(
            f"Unsupported language: {lang}. Only 'python' and 'cpp' are supported."
        )
    
    code_bytes = code.encode('utf-8')
    tree = parser.parse(code_bytes)
    root_node = tree.root_node
    return root_node, code_bytes

def get_tokens(node, code_bytes, max_depth: int = 1000):
    tokens = []
    stack = [(node, 0)]
    
    while stack:
        current_node, depth = stack.pop()
        if depth >= max_depth: break # STOP
        
        if not current_node.children:
            token_text = code_bytes[current_node.start_byte : current_node.end_byte].decode('utf-8')
            token_prefix = code_bytes[:current_node.start_byte].decode('utf-8')
            tokens.append({
                'type': current_node.type,
                'token': token_text,
                'prefix': token_prefix,
                'depth': depth
            })
        else:
            for child in reversed(current_node.children):
                stack.append((child, depth + 1))
    
    return tokens

def code_parse(code: str, lang: str = 'python'):
    try: 
        root_node, code_bytes = get_root_node(code, lang)
        tokens = get_tokens(root_node, code_bytes)
        return tokens
    except ValueError as e:
        print("Error:", e)

test_code = """
import numpy as np

def generate_combinations(vertices, depths, all_indices):
    combs = []
    for indices in all_indices:
        comb = []
        for vtx, i in zip(vertices, indices):
            comb.append(np.array([vtx[0], vtx[1], depths[i]]))
        combs.append(comb)
    return combs

# Test the function
vertices = [[1, 2], [3, 4], [5, 6]]
depths = [10, 20, 30]
all_indices = [[0, 1], [1, 2]]
result = generate_combinations(vertices, depths, all_indices)
print(result)  # Output: [[array([1, 2, 10]), array([3, 4, 20])], [array([3, 4, 20]), array([5, 6, 30])]]
if x <= y:
   x = x + 1
"""

if __name__ == "__main__":
    root_node, code_bytes = get_root_node(test_code)
    tokens = get_tokens(root_node, code_bytes)
    
    token_results = [item['token'] for item in tokens]
    type_results = [item['type'] for item in tokens]
    
    print(token_results)
    print(type_results)