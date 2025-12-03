import hashlib
import os
import re
import json
import yaml
from typing import Optional, Tuple, List, Dict
import argparse
import libcst as cst
import jinja2
from libcst import CSTTransformer, parse_module, parse_expression
from .blocks import compound
from .blocks import module

# Just so we don't have to even import litellm (slow) if we have no changes
_text_completion = None
def text_completion(*arg, **kw):
    global _text_completion
    if _text_completion is None:
        from litellm import text_completion as _text_completion
    return _text_completion(*arg, **kw)

builtin_config = {
    "base": {
        "model": "openai/gpt-4",
    },

    "FunctionDef": {
        "prompt": """
Generate the python source code for a function with the following
signature, docstring, and initial comments.

{{signature}}

# IMPORTANT
 * Write the full function implementation.
 * Provide only valid python for a single function as output.
 * Do NOT add any initial description, argument or similar
"""
    },
    
    "ClassDef": {
        "prompt": """
Below is the python source code for a class and some of its methods
(without implementations). Given the docstring and initial comments of
the class, define any missing method signatures and provide their
docstrings.

{{signature}}

# IMPORTANT
 * Write the full class specification.
 * Provide only valid skeleton python for a single class as output.
 * Do NOT add any initial description or similar
"""
    }
}

def sha_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def sha_hash_data(data):
    return sha_hash(
        json.dumps(
            data,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False
        ))
    
def stringify(node):
    return cst.Module([node]).code_for_node(node)

class TransformCode(CSTTransformer):
    @classmethod
    def transform_code(cls, src, **kw):
        return parse_module(src).visit(cls(**kw)).code

    def __init__(self, config = {}):
        self.config = config                 
        self.imports: Set[cst.SimpleStatementLine] = set()

    def get_config(self, config_path):
        return merge_configs(
            *[self.config.get(key)
              for key in ["base"] + config_path])
        
    def leave_FunctionDef(self, original_node, updated_node):
        replacement = self.replace_FunctionDef(original_node)
        if replacement is None:
            return updated_node
        return replacement
        
    def generate_FunctionDef(self, signature):
        conf = self.get_config(["FunctionDef"])

        prompt = jinja2.Template(conf["prompt"]).render(signature = stringify(signature), **conf)

        response = text_completion(model=conf["model"], prompt=prompt)
        llm_result = response.choices[0].text.strip()
        
        if "```python" in llm_result:
            llm_result = llm_result.split("```python", 1)[1]
            llm_result = llm_result.rsplit("```", 1)[0]
        elif "```" in llm_result:
            llm_result = llm_result.split("```", 1)[1]
            llm_result = llm_result.rsplit("```", 1)[0]
        try:
            return cst.parse_module(llm_result)
        except Exception as e:
            raise Exception("Unable to parse: %s\n%s" % (e, llm_result)) from e

    def signature_codeid(self, sig, config_path = []):
        config_hash = sha_hash_data(self.get_config(config_path))
        
        data = {"generate": False, "autogen": False, "metadata": {}, **sig}

        old_codeid = data["metadata"].get("codeid", None)
        signature_str = stringify(data["signature"]).strip()

        assert "# LITERALAI: " not in signature_str, "Parsing problem..."
        
        new_codeid = sha_hash(config_hash + signature_str)
        
        if (    (new_codeid != old_codeid)
            and (   old_codeid is not None
                 or data["body"] is None)):
            data["generate"] = True            

        if new_codeid == data["metadata"].get("genid", None):
            data["autogen"] = True

        data["old_codeid"] = old_codeid
        data["codeid"] = new_codeid
        data["metadata"]["codeid"] = new_codeid
        
        return data
    
    def replace_FunctionDef(self, node):
        leading_lines = node.leading_lines
        node = node.with_changes(leading_lines = [])

        sig = self.signature_codeid(compound.extract_metadata(node), ["FunctionDef"])
        if not sig["generate"]:
            return None        

        print("===={generate function}====")
        print("----{signature}----")
        print(stringify(sig["signature"]))
        print("----{hash}----")
        print("Old:", sig["old_codeid"])
        print("New:", sig["codeid"])
        
        llm_result = self.generate_FunctionDef(sig["signature"])
        
        #print("===={llm}====")
        #print(llm_result)

        llm_code = module.extract_imports_and_def(llm_result)
        self.imports.update(llm_code["imports"])
        
        new_sig = compound.extract_signature(llm_code["definition"])

        replacement = compound.unextract_metadata({**sig, "body": new_sig["body"]})
        
        print("----{imports}----")
        print("\n".join([stringify(imp) for imp in llm_code["imports"]]))
        print("----{function}----")
        print(stringify(replacement))
        
        return replacement.with_changes(leading_lines = leading_lines)
    
    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.CSTNode:
        replacement = self.replace_ClassDef(original_node)
        if replacement is None:
            return updated_node
        return replacement
    
    def generate_ClassDef(self, signature):
        conf = self.get_config(["ClassDef"])
        prompt = jinja2.Template(conf["prompt"]).render(signature = stringify(signature["signature"]), **conf)

        response = text_completion(model=conf["model"], prompt=prompt)
        llm_result = response.choices[0].text.strip()

        if "```python" in llm_result:
            llm_result = llm_result.split("```python", 1)[1]
            llm_result = llm_result.rsplit("```", 1)[0]

        return cst.parse_module(llm_result)
            
    def replace_ClassDef(self, node):
        leading_lines = node.leading_lines
        node = node.with_changes(leading_lines = [])

        sig = self.signature_codeid(compound.extract_metadata(node), ["ClassDef"])
        if not sig["generate"]:
            return None

        if sig["body"] is None:
            llm_sig = sig
        else:
            manual = []
            for stmt in sig["body"].body:
                if isinstance(stmt, cst.FunctionDef):
                    method_sig = self.signature_codeid(compound.extract_metadata(stmt), ["FunctionDef"])
                    if method_sig["autogen"]:
                        continue
                manual.append(stmt)
            llm_sig = {**sig,
                       "signature": compound.append_body(
                           sig["signature"],
                           sig["body"].with_changes(body = manual)),
                       "body": None}
                                         
        print("===={generate class}====")
        print("----{signature}----")
        print(stringify(llm_sig["signature"]))
        print("----{hash}----")
        print("Old:", sig["old_codeid"])
        print("New:", sig["codeid"])

        llm_result = self.generate_ClassDef(llm_sig)
        print("----{llm}----")
        print(stringify(llm_result))

        llm_code = module.extract_imports_and_def(llm_result)
        self.imports.update(llm_code["imports"])

        llm_result_sig = compound.extract_signature(llm_code["definition"])

        with_metadata = []
        for stmt in llm_result_sig["body"].body:
            if isinstance(stmt, cst.FunctionDef):
                method_sig = self.signature_codeid(compound.extract_signature(stmt), ["FunctionDef"])
                # Generate the body in a recursive call to the LLM!
                method_sig["metadata"]["genid"] = method_sig["metadata"].pop("codeid")
                method_sig["body"] = None
                stmt = compound.unextract_metadata(method_sig)
            with_metadata.append(stmt)
        llm_result_sig["body"] = llm_result_sig["body"].with_changes(
            body = with_metadata)

        llm_result_sig["body"] = llm_result_sig["body"].visit(self)
        
        replacement = compound.unextract_metadata(
            {**sig, "body": compound.append_blocks(sig["body"], llm_result_sig["body"])})

        print("----{class}----")
        print(stringify(replacement))
        
        return replacement.with_changes(leading_lines = leading_lines)
    
    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module):
        return updated_node.with_changes(
            body=list(self.imports) + list(updated_node.body)
        )    
    
def process_file(filepath: str, config = {}):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    new_source = TransformCode.transform_code(source, config=config)
    # print("===={new source}====")
    # print(new_source)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_source)

def merge_config(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: merge_config(a[k], b[k]) if (k in a and k in b) else a.get(k, b.get(k))
                for k in set(a.keys()).union(b.keys())}
    else:
        return b
        
def merge_configs(*configs):
    if len(configs) == 1:
        return configs[0]
    return merge_config(configs[0], merge_configs(*configs[1:]))
        
def process_directory(root_dir: str):
    root_dir = os.path.abspath(root_dir)
    stack = []  # stack of (dirpath, literalai.yml path or None)
    prev_depth = 0

    for dirpath, _, filenames in os.walk(root_dir):
        rel = os.path.relpath(dirpath, root_dir)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        while len(stack) > depth:
            stack.pop()
        cfg = {}
        if "literalai.yml" in filenames:
            with open(os.path.join(dirpath, "literalai.yml")) as f:
                cfg = yaml.safe_load(f)
        stack.append((dirpath, cfg))
        config_files = [c for _, c in stack if c]

        for fname in filenames:
            if fname.endswith(".py"):
                filepath = os.path.join(dirpath, fname)
                print(f"Processing {filepath}")
                process_file(filepath, merge_configs(*([builtin_config] + config_files)))

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
