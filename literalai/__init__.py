import hashlib
import os
import re
from typing import Optional, Tuple, List, Dict
import argparse
import litellm
from litellm import text_completion
import libcst as cst
from libcst import CSTTransformer, parse_module, parse_expression
from .extract import extract_fn_signature
from .indent import set_indent

CODEID_PATTERN = re.compile(r"#\s*CODEID=([0-9a-f]+)")

def sha_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

class TransformCode(CSTTransformer):
    @classmethod
    def transform_code(cls, src):
        return parse_module(src).visit(cls()).code

    
    
    def leave_FunctionDef(self, original_node, updated_node):
        replacement = self.replace_FunctionDef(original_node)
        if replacement is None:
            return updated_node
        return replacement
    
    def generate_FunctionDef(self, signature):
        prompt = f"""
Generate the python source code for a function with the following
signature, docstring, and initial comments.

{signature}
        
Provide only valid python code for the function body. Do not include
function signature, the docstring or comments given above in the
output.
"""

        print("===={prompt}====")
        print(prompt)
        print("===={/prompt}====")
        response = text_completion(model="openai/gpt-4", prompt=prompt)
        return response.choices[0].text
    
    def replace_FunctionDef(self, node):
        signature, indent = extract_fn_signature(node)
        old_codeid = None
        if "CODEID:" in signature[-1]:
            old_codeid = signature[-1].split("CODEID:")[1].strip()
            signature = signature[:-1]
        signature = "\n".join(signature)
        
        codeid = sha_hash(signature)
        if codeid == old_codeid:
            return None

        if not signature.strip():
            import pdb
            pdb.set_trace()
        
        
        new_body = set_indent(self.generate_FunctionDef(signature), indent)

        replacement = f"{signature}\n{indent}# CODEID:{codeid}\n{new_body}"

        print("===={generate}====")
        print(replacement)
        
        return cst.parse_statement(
            replacement)
    
    def replace_cls(self, node):
        return None

def process_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    new_source = TransformCode.transform_code(source)
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
    



# def regenerate_class(node: ast.ClassDef, atok: asttokens.ASTTokens) -> str:
#     # --- Step 1: Collect class info ---
#     class_sig_doc = concat_signature_doc_comment(node, atok)
    
#     # --- Step 2: Collect existing methods ---
#     methods_info: List[Dict] = []
#     for n in node.body:
#         if isinstance(n, ast.FunctionDef):
#             sig_doc = concat_signature_doc_comment(n, atok)
#             methods_info.append({
#                 "node": n,
#                 "sig_doc": sig_doc
#             })
    
#     methods_str = "\n\n".join(m['sig_doc'] for m in methods_info)
    
#     # --- Step 3: Prepare prompt for LLM ---
#     prompt = f"""
# Regenerate the Python class given its signature, docstring, comments, and methods.
# - Include CODEID comments for each method.
# - Generate any missing methods (signatures and docstrings).
# - If a method should be deleted, mark it for deletion.
# - If a method should be changed, provide new signature, docstring, or comments,
# but do not change methods unnecessarily.

# "Class:
# {class_sig_doc}

# Methods:
# {methods_str}

# Return structured output:

# {
#   "add": ["new signature + docstring + comments"],
#   "delete": ["old_method"],
#   "update": {
#       "existing_method": "new signature + docstring + comments",
#       "another_method": "new signature + docstring + comments"
#   }
# }
# """
    
#     # --- Step 4: Call LLM ---
#     response = text_completion(model="openai/gpt-4", prompt=prompt)
#     llm_result = parse_llm_output(response.choices[0].text)
    
#     # --- Step 6: Update the AST ---
#     # Delete methods
#     node.body = [
#         n for n in node.body
#         if not (isinstance(n, ast.FunctionDef) and n.name in llm_result.get("delete", []))
#     ]
    
#     # Update methods
#     for method_name, new_code in llm_result.get("update", {}).items():
#         for i, n in enumerate(node.body):
#             if isinstance(n, ast.FunctionDef) and n.name == method_name:
#                 new_node = ast.parse(new_code).body[0]
#                 # Preserve CODEID comments if present
#                 new_node.body[0:0] = n.body[:1] if n.body and isinstance(n.body[0], ast.Expr) and hasattr(n.body[0], 'value') else []
#                 node.body[i] = new_node
    
#     # Add new methods
#     existing_method_names = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
#     for method_name, new_code in llm_result.get("add", {}).items():
#         if method_name not in existing_method_names:
#             node.body.append(ast.parse(new_code).body[0])
    
#     # --- Step 7: Recursively regenerate method bodies ---
#     for n in node.body:
#         if isinstance(n, ast.FunctionDef):
#             new_code, _ = process_node(n, atok)
#             # Replace method body with regenerated body
#             n.body = ast.parse(new_code).body[0].body
    
#     # --- Step 8: Convert AST back to code ---
#     return atok.get_text(node)
