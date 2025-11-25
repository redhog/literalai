import ast
import asttokens
import hashlib
import os
import re
from typing import Optional, Tuple, List, Dict

# LLM imports
import litellm
from litellm import OpenAI

# Initialize LLM
llm = OpenAI(model="gpt-4")

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
        if tok.type == asttokens.tokenize.COMMENT:
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
    response = llm.generate(prompt)
    return response.strip()


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
    response = llm.generate(prompt)
    llm_result = parse_llm_output(response)
    
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
    """Returns new code for node and new codeid."""
    sig_doc_comment = concat_signature_doc_comment(node, atok)
    # Extract existing CODEID
    tokens = list(atok.get_tokens(node))
    codeid_token = None
    for tok in reversed(tokens):
        if tok.type == asttokens.tokenize.COMMENT and CODEID_PATTERN.search(tok.string):
            codeid_token = tok.string
            break
    existing_codeid = get_codeid_from_comment(codeid_token)
    calculated_hash = sha_hash(sig_doc_comment)
    if existing_codeid == calculated_hash:
        # No change needed
        return atok.get_text(node), existing_codeid
    # Regenerate
    if isinstance(node, ast.FunctionDef):
        new_code = regenerate_function(sig_doc_comment)
    elif isinstance(node, ast.ClassDef):
        new_code = regenerate_class(node, atok)
    else:
        return atok.get_text(node), existing_codeid
    # Append new CODEID
    new_code += f"\n#CODEID={calculated_hash}"
    return new_code, calculated_hash

def process_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    atok = asttokens.ASTTokens(source, parse=True)
    tree = atok.tree
    new_source = source
    # We will process top-level nodes only
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            new_code, new_codeid = process_node(node, atok)
            # Replace old code with new code
            start, end = atok.get_text_range(node)
            new_source = new_source[:start] + new_code + new_source[end:]
    # Write back
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_source)

def process_directory(root_dir: str):
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".py"):
                filepath = os.path.join(dirpath, fname)
                print(f"Processing {filepath}")
                process_file(filepath)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate Python functions/classes with docstring verification.")
    parser.add_argument("source_dir", help="Root directory of Python source code")
    args = parser.parse_args()
    process_directory(args.source_dir)
