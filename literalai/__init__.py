import hashlib
import os
import re
import json
from typing import Optional, Tuple, List, Dict
import argparse
import litellm
from litellm import text_completion
import libcst as cst
from libcst import CSTTransformer, parse_module, parse_expression
from .extract import extract_fn_signature, extract_cls_signature, ExtractImportsFn
from .indent import set_indent

CODEID_PATTERN = re.compile(r"#\s*CODEID=([0-9a-f]+)")

def sha_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

class TransformCode(CSTTransformer):
    @classmethod
    def transform_code(cls, src):
        return parse_module(src).visit(cls()).code

    def __init__(self):
        self.imports: Set[cst.SimpleStatementLine] = set()

    
    def leave_FunctionDef(self, original_node, updated_node):
        replacement = self.replace_FunctionDef(original_node)
        if replacement is None:
            return updated_node
        return replacement
    
    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.CSTNode:
        replacement = self.replace_ClassDef(original_node)
        if replacement is None:
            return updated_node
        return replacement
    
    def generate_FunctionDef(self, signature):
        prompt = f"""
Generate the python source code for a function with the following
signature, docstring, and initial comments.

{signature}

# IMPORTANT
 * Write the full function implementation.
 * Provide only valid python for a single function as output.
 * Do NOT add any initial description, argument or similar
"""

        response = text_completion(model="openai/gpt-4", prompt=prompt)
        llm_result = response.choices[0].text.strip()
        
        if "```python" in llm_result:
            llm_result = llm_result.split("```python", 1)[1]
            llm_result = llm_result.rsplit("```", 1)[0]

        return llm_result

    def signature_codeid(self, sig):
        data = {"generate": False, "autogen": False, "metadata": {}, **sig}
        if "LITERALAI:" in data["signature"][-1]:
            data["metadata"] = json.loads(data["signature"][-1].split("LITERALAI:")[1].strip())
            data["signature"] = data["signature"][:-1]
        data["signature"] = "\n".join(data["signature"])

        old_codeid = data["metadata"].get("codeid", None)
        new_codeid = sha_hash(data["signature"])
        
        if (    (new_codeid != old_codeid)
            and (   old_codeid is not None
                 or not "".join(data["body"]).strip())):
            data["generate"] = True

        if new_codeid == data["metadata"].get("genid", None):
            data["autogen"] = True

        data["old_codeid"] = old_codeid
        data["codeid"] = new_codeid
        data["metadata"]["codeid"] = new_codeid
        
        return data
    
    def replace_FunctionDef(self, node):
        sig = self.signature_codeid(extract_fn_signature(node))
        if not sig["generate"]:
            return None        

        print("===={generate function}====")
        print("----{signature}----")
        print(sig["signature"])
        print("----{hash}----")
        print("Old:", sig["old_codeid"])
        print("New:", sig["codeid"])
        
        llm_result = self.generate_FunctionDef(sig["signature"])
        
        #print("===={llm}====")
        #print(llm_result)

        llm_code = ExtractImportsFn.run(llm_result)
        self.imports.update(llm_code["imports"])
        
        new_sig = extract_fn_signature(llm_code["fn"])
        new_body = '\n'.join(new_sig["body"])
        replacement = f"{sig['signature']}\n{sig['indentation']}# LITERALAI:{json.dumps(sig['metadata'])}\n{new_body}"

        print("----{imports}----")
        print("\n".join([cst.Module([imp]).code_for_node(imp) for imp in llm_code["imports"]]))
        print("----{function}----")
        print(replacement)
        
        return cst.parse_statement(
            replacement)
    
    def generate_ClassDef(self, signature, method_signatures):
        method_signatures = "\n\n".join(msig["signature"] for msig in method_signatures)
        prompt = f"""
Below is the python source code for a class and some of its methods
(without implementations). Given the docstring and initial comments of
the class, define any missing method signatures and provide their
docstrings.

{signature["signature"]}

{method_signatures}
        

# IMPORTANT
 * Write the full class specification.
 * Provide only valid skeleton python for a single class as output.
 * Do NOT add any initial description or similar
"""

        response = text_completion(model="openai/gpt-4", prompt=prompt)
        llm_result = response.choices[0].text.strip()

        if "```python" in llm_result:
            llm_result = llm_result.split("```python", 1)[1]
            llm_result = llm_result.rsplit("```", 1)[0]

        return llm_result
            
    def replace_ClassDef(self, node):
        sig = self.signature_codeid(extract_cls_signature(node))
        if not sig["generate"]:
            return None        

        methods = [stmt for stmt in node.body.body if isinstance(stmt, cst.FunctionDef)]
        methods_signatures = [self.signature_codeid(extract_fn_signature(method)) for method in methods]
        other_methods_signatures = [msig for msig in methods_signatures if not sig["autogen"]]
        updatable_methods_signatures = [msig for msig in methods_signatures if sig["autogen"]]

        
        print("===={generate class}====")
        print("----{signature}----")
        print(sig["signature"] + "\n\n" + "\n\n".join(sig["signature"] for sig in other_methods_signatures))
        print("----{hash}----")
        print("Old:", sig["old_codeid"])
        print("New:", sig["codeid"])

        new_src = self.generate_ClassDef(sig, other_methods_signatures)
        print("----{llm}----")
        print(new_src)
        new_sig = extract_cls_signature(parse_module(new_src))

        new_body = '\n'.join(new_sig['body'])
        
        replacement_node = cst.parse_module(f"{sig['signature']}\n{sig['indentation']}# LITERALAI:{json.dumps(sig['metadata'])}\n{new_body}").body[0]
        
        replacement_node = replacement_node.with_changes(
            body=replacement_node.body.with_changes(
                body=list(replacement_node.body.body) + [msig["node"] for msig in other_methods_signatures]
            ))

        print("----{class}----")
        print(cst.Module([replacement_node]).code_for_node(replacement_node))
        
        return replacement_node
    
    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module):
        return updated_node.with_changes(
            body=list(self.imports) + list(updated_node.body)
        )    
    
def process_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    new_source = TransformCode.transform_code(source)
    # print("===={new source}====")
    # print(new_source)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_source)

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
