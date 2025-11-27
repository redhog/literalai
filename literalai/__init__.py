import ast
import asttokens
import hashlib
import os
import re
from typing import Optional, Tuple, List, Dict
import argparse
import tokenize

# LLM imports
import litellm
from litellm import text_completion

CODEID_PATTERN = re.compile(r"#\s*CODEID=([0-9a-f]+)")

def sha_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_codeid_from_comment(comment: Optional[str]) -> Optional[str]:
    if not comment:
        return None
    match = CODEID_PATTERN.search(comment)
    if match:
        return match.group(1)
    return None

def extract_comments_before(node, atok: asttokens.ASTTokens) -> str:
    """Extract initial comments before a function/class definition."""
    tokens = atok.get_tokens(node)
    comments = []
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            if not CODEID_PATTERN.search(tok.string):
                comments.append(tok.string.lstrip("#").strip())
    return "\n".join(comments)

def get_docstring(node: ast.AST) -> str:
    return ast.get_docstring(node) or ""

def concat_signature_doc_comment(node: ast.AST, atok: asttokens.ASTTokens) -> str:
    if isinstance(node, ast.FunctionDef):
        args = [arg.arg for arg in node.args.args]
        sig = f"def {node.name}({', '.join(args)})"
        if node.returns:
            sig += f" -> {ast.unparse(node.returns)}"
    elif isinstance(node, ast.ClassDef):
        bases = [ast.unparse(base) for base in node.bases]
        sig = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
    else:
        sig = ""
    comments = extract_comments_before(node, atok)
    docstring = get_docstring(node)
    return "\n".join([sig, docstring, comments]).strip()

def regenerate_function(signature_doc_comment: str) -> str:
    prompt = (
        "Regenerate the Python function using this signature, docstring, and comments. "
        "Do NOT include the original function body.\n\n"
        f"{signature_doc_comment}\n\n"
        "Provide only valid Python code."
    )
    response = text_completion(model="openai/gpt-4", prompt=prompt)
    return response.choices[0].text.strip()


def regenerate_class(node: ast.ClassDef, atok: asttokens.ASTTokens) -> str:
    # --- Step 1: Collect class info ---
    class_sig_doc = concat_signature_doc_comment(node, atok)
    
    # --- Step 2: Collect existing methods ---
    methods_info: List[Dict] = []
    for n in node.body:
        if isinstance(n, ast.FunctionDef):
            sig_doc = concat_signature_doc_comment(n, atok)
            methods_info.append({
                "node": n,
                "sig_doc": sig_doc
            })
    
    methods_str = "\n\n".join(m['sig_doc'] for m in methods_info)
    
    # --- Step 3: Prepare prompt for LLM ---
    prompt = f"""
Regenerate the Python class given its signature, docstring, comments, and methods.
- Include CODEID comments for each method.
- Generate any missing methods (signatures and docstrings).
- If a method should be deleted, mark it for deletion.
- If a method should be changed, provide new signature, docstring, or comments,
but do not change methods unnecessarily.

"Class:
{class_sig_doc}

Methods:
{methods_str}

Return structured output:

{
  "add": ["new signature + docstring + comments"],
  "delete": ["old_method"],
  "update": {
      "existing_method": "new signature + docstring + comments",
      "another_method": "new signature + docstring + comments"
  }
}
"""
    
    # --- Step 4: Call LLM ---
    response = text_completion(model="openai/gpt-4", prompt=prompt)
    llm_result = parse_llm_output(response.choices[0].text)
    
    # --- Step 6: Update the AST ---
    # Delete methods
    node.body = [
        n for n in node.body
        if not (isinstance(n, ast.FunctionDef) and n.name in llm_result.get("delete", []))
    ]
    
    # Update methods
    for method_name, new_code in llm_result.get("update", {}).items():
        for i, n in enumerate(node.body):
            if isinstance(n, ast.FunctionDef) and n.name == method_name:
                new_node = ast.parse(new_code).body[0]
                # Preserve CODEID comments if present
                new_node.body[0:0] = n.body[:1] if n.body and isinstance(n.body[0], ast.Expr) and hasattr(n.body[0], 'value') else []
                node.body[i] = new_node
    
    # Add new methods
    existing_method_names = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
    for method_name, new_code in llm_result.get("add", {}).items():
        if method_name not in existing_method_names:
            node.body.append(ast.parse(new_code).body[0])
    
    # --- Step 7: Recursively regenerate method bodies ---
    for n in node.body:
        if isinstance(n, ast.FunctionDef):
            new_code, _ = process_node(n, atok)
            # Replace method body with regenerated body
            n.body = ast.parse(new_code).body[0].body
    
    # --- Step 8: Convert AST back to code ---
    return atok.get_text(node)


def process_node(node: ast.AST, atok: asttokens.ASTTokens) -> Tuple[str, str]:
    """Returns new code for node."""
    
    if not isinstance(node, ast.FunctionDef) and not isinstance(node, ast.ClassDef):
        return None
    
    sig_doc_comment = concat_signature_doc_comment(node, atok)
    tokens = list(atok.get_tokens(node))
    codeid_token = None
    for tok in reversed(tokens):
        if tok.type == tokenize.COMMENT and CODEID_PATTERN.search(tok.string):
            codeid_token = tok.string
            break
    existing_codeid = get_codeid_from_comment(codeid_token)
    calculated_hash = sha_hash(sig_doc_comment)
    if existing_codeid == calculated_hash:
        return None
    
    if isinstance(node, ast.FunctionDef):
        new_code = regenerate_function(sig_doc_comment)
    elif isinstance(node, ast.ClassDef):
        new_code = regenerate_class(node, atok)
    else:
        return atok.get_text(node), existing_codeid
    new_code += f"\n#CODEID={calculated_hash}"
    return new_code

def transform_source(source, process_node):
    """
    Recursively walk the AST of `source`, applying `process_node` to each node.
    If `process_node(node)` returns a string, replace that node in the source
    with the returned code snippet and update AST positions.
    
    Returns the transformed source code.
    """
    atok = asttokens.ASTTokens(source, parse=True)
    root = atok.tree

    def recurse(node):
        nonlocal atok, root, source

        # Try to process the current node
        new_code = process_node(node, atok)
        if new_code is not None:
            # Replace the code in the original source
            start, end = atok.get_text_range(node, atok)
            old_source = source
            source = source[:start] + new_code + source[end:]

            print("===={old}====")
            print(old_source)
            print("===={new}====")
            print(source)
            print()
            print()
            
            # Re-parse the updated source to update AST positions
            try:
                atok = asttokens.ASTTokens(source, parse=True)
            except Exception as e:
                raise Exception("%s:\n================\n%s\n================\n" % (e, source)) from e
            
            root = atok.tree
            
            # Return the new node corresponding to the inserted code
            new_node = ast.parse(new_code).body[0]
            atok.mark_tokens(new_node)  # Ensure new node has token positions
            return new_node
        
        # Recurse into child nodes
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        new_item = recurse(item)
                        if new_item is not None:
                            value[i] = new_item
            elif isinstance(value, ast.AST):
                new_value = recurse(value)
                if new_value is not None:
                    setattr(node, field, new_value)
        return None

    recurse(root)
    return source

def process_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    new_source = transform_source(source, process_node)
    print("===={new source}====")
    print(new_source)
    #with open(filepath, "w", encoding="utf-8") as f:
    #    f.write(new_source)

def process_directory(root_dir: str):
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".py"):
                filepath = os.path.join(dirpath, fname)
                print(f"Processing {filepath}")
                process_file(filepath)

def main():
    parser = argparse.ArgumentParser(description="Regenerate Python functions/classes with docstring verification.")
    parser.add_argument("source_dir", help="Root directory of Python source code")
    args = parser.parse_args()
    process_directory(args.source_dir)
                
if __name__ == "__main__":
    main()
    
