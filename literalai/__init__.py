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
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

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

    def __init__(self, config = {}, **kw):
        self.config = config                 
        self.imports: Set[cst.SimpleStatementLine] = set()
        self.options = kw
        
    def get_config(self, config_path):
        return merge_configs(
            *[self.config.get(key)
              for key in ["base"] + config_path])
        
    def leave_FunctionDef(self, original_node, updated_node):
        replacement = self.replace_FunctionDef(updated_node)
        if replacement is None:
            return updated_node
        return replacement
        
    def generate(self, sig, conf):
        response = text_completion(model=conf["model"], prompt=sig["prompt"])
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

    def get_prompt(self, sig, conf):
        return {
            "prompt": jinja2.Template(conf["prompt"]).render(
                signature = stringify(sig["signature"]),
                **conf),
            **sig}
        
    def update_metadata(self, sig, conf, autogen=False):
        config_hash = sha_hash_data(conf)
        
        data = {"generate": False, "autogen": False, "metadata": {}, **sig}

        old_prompt = data["metadata"].get("prompt", None)
        old_gen = data["metadata"].get("gen", None)

        assert "# LITERALAI: " not in sig["prompt"], "Parsing problem: %s" % signature_str
        
        new_prompt = sha_hash(config_hash + sig["prompt"])
        new_gen = sha_hash(stringify(sig["signature"]))

        if (    (new_prompt != old_prompt)
            and (   old_prompt is not None
                 or data["body"] is None)):
            data["generate"] = True            

        if new_gen == data["metadata"].get("gen", None):
            data["autogen"] = True

        data["old_promptid"] = old_prompt
        data["old_gen"] = old_gen
        data["promptid"] = new_prompt
        data["metadata"]["prompt"] = new_prompt
        if autogen:
            data["metadata"]["gen"] = new_gen

        return data

    
    def replace_FunctionDef(self, node):
        leading_lines = node.leading_lines
        node = node.with_changes(leading_lines = [])

        conf = self.get_config(["FunctionDef"])
        sig = self.get_prompt(compound.extract_metadata(node), conf)
        
        sig = self.update_metadata(sig, conf)
        if not sig["generate"]:
            return None        

        print("===={generate function}====")
        print("----{signature}----")
        print(stringify(sig["signature"]))
        print("----{hash}----")
        print("Old:", sig["old_promptid"])
        print("New:", sig["promptid"])

        if self.options.get("super_dry_run"):
            return node
        
        llm_result = self.generate(sig, conf)
        
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
        replacement = self.replace_ClassDef(updated_node)
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
        
        conf = self.get_config(["ClassDef"])
        fn_conf = self.get_config(["FunctionDef"])
        sig = compound.extract_metadata(node)

        if sig["body"] is None:
            llm_sig = sig
        else:
            manual = []
            for stmt in sig["body"].body:
                if isinstance(stmt, cst.FunctionDef):
                    method_sig = self.update_metadata(
                        self.get_prompt(
                            compound.extract_metadata(stmt),
                            fn_conf),
                        fn_conf)
                    if method_sig["autogen"]:
                        continue
                    stmt = method_sig["signature"]
                manual.append(stmt)
            llm_sig = {**sig,
                       "signature": compound.append_body(
                           sig["signature"],
                           sig["body"].with_changes(body = manual)),
                       # Just for update_metadata to get generate() right
                       "body": sig["body"].with_changes(body = manual)}

        llm_sig = self.get_prompt(llm_sig, conf)
        llm_sig = self.update_metadata(llm_sig, conf)
        if not llm_sig["generate"]:
            return None
            
        print("===={generate class}====")
        print("----{signature}----")
        print(stringify(llm_sig["signature"]))
        print("----{hash}----")
        print("Old:", llm_sig["old_promptid"])
        print("New:", llm_sig["promptid"])

        if self.options.get("super_dry_run"):
            return node
        
        llm_result = self.generate(llm_sig, conf)
        print("----{llm}----")
        print(stringify(llm_result))

        llm_code = module.extract_imports_and_def(llm_result)
        self.imports.update(llm_code["imports"])

        llm_result_sig = compound.extract_signature(llm_code["definition"])

        with_metadata = []
        for stmt in llm_result_sig["body"].body:
            if isinstance(stmt, cst.FunctionDef):
                method_sig = self.update_metadata(
                    self.get_prompt(
                        compound.extract_metadata(stmt),
                        fn_conf),
                    fn_conf,
                    autogen = True)

                # Generate the body in a recursive call to the LLM!
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
    
def process_file(filepath: str, config = {}, **kw):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    new_source = TransformCode.transform_code(source, config=config, **kw)
    if kw.get("dry_run"):
        print("===={new source}====")
        print(new_source)
    if kw.get("dry_run") or kw.get("super_dry_run"):
        return
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
        
def process_directory(root_dir: str, **kw):
    root_dir = os.path.abspath(root_dir)
    stack = []  # stack of (dirpath, literalai.yml path or None)
    prev_depth = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Don't go into .git etc
        for name in list(dirnames):
            if name.startswith("."):
                dirnames.remove('.git')
            
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
                process_file(filepath, merge_configs(*([builtin_config] + config_files)), **kw)

def main():
    parser = argparse.ArgumentParser(
        description="""Generate implementations for Python functions
and classes with only docstrings, and regenerate previously generated
functions and classes where the docstrings have changed.""")
    parser.add_argument(
        "source_dir",
        nargs="?",
        default=".",
        help="Root directory of Python source code (default: current directory)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write the updated code to disk, just print it to screen."
    )
    parser.add_argument(
        "--super-dry-run",
        action="store_true",
        help="Do not update code, just show what would be updated. This does not do any LLM calls."
    )

    args = parser.parse_args()
    process_directory(args.source_dir, dry_run=args.dry_run, super_dry_run=args.super_dry_run)
                
if __name__ == "__main__":
    main()
