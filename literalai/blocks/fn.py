import libcst as cst
import json
from typing import Tuple, Dict

import libcst as cst
import json
from typing import Tuple, Dict

import libcst as cst
from typing import Optional

def extract_metadata(func_node: cst.FunctionDef) -> Tuple[Dict, bool]:
    """
    Extract JSON from a `# LITERALAI:` comment in a function body.

    Returns:
        - The parsed JSON as a dict, or {} if not found.
        - Boolean indicating whether the function has real statements.
    """
    literalai_data = {}
    has_real_statements = False

    # Check if there are real statements
    if func_node.body.body:
        # Filter out just docstrings and pass (EmptyLine) statements
        has_real_statements = any(
            not isinstance(stmt, (cst.SimpleStatementLine, cst.Pass)) or
            not (len(stmt.body) == 1 and isinstance(stmt.body[0], cst.Expr) and isinstance(stmt.body[0].value, cst.SimpleString))
            for stmt in func_node.body.body
        )

    # Collect all comments: both inline and footer (trailing) comments
    comments = []

    for stmt in func_node.body.body:
        # Inline comments
        if isinstance(stmt, cst.SimpleStatementLine):
            for comment in stmt.leading_lines:
                if comment.comment:
                    comments.append(comment.comment.value)
            for small_stmt in stmt.body:
                if hasattr(small_stmt, "trailing_whitespace"):
                    trailing_comment = getattr(small_stmt.trailing_whitespace, "comment", None)
                    if trailing_comment:
                        comments.append(trailing_comment.value)

        # EmptyLine with comment (used for footer)
        if hasattr(stmt, "leading_lines"):
            for line in stmt.leading_lines:
                if line.comment:
                    comments.append(line.comment.value)

    # Footer comments are also stored in the block footer
    if hasattr(func_node.body, "footer"):
        for line in func_node.body.footer:
            if isinstance(line, cst.EmptyLine) and line.comment:
                comments.append(line.comment.value)

    # Search for LITERALAI comment
    for comment in comments:
        if comment.strip().startswith("# LITERALAI:"):
            json_text = comment.strip()[len("# LITERALAI:"):].strip()
            try:
                literalai_data = json.loads(json_text)
            except json.JSONDecodeError:
                literalai_data = {}
            break

    return {"metadata": literalai_data,
            "empty": not has_real_statements}


def extract_signature(func_node: cst.FunctionDef) -> cst.FunctionDef:
    comments = []
    stmts = []
    found_metadata = False

    stmt  = None
    for stmt in func_node.body.body:
        if (    not isinstance(stmt, cst.Pass)
            and not (    hasattr(stmt, "body")
                     and len(stmt.body) == 1
                     and isinstance(stmt.body[0], cst.Expr)
                     and isinstance(stmt.body[0].value, cst.SimpleString))):
            for comment in stmt.leading_lines:
                if comment.comment:
                    if "# LITERALAI:" in comment.comment.value:
                        found_metadata = True
                        break
                    comments.append(
                        cst.EmptyLine(
                            indent=True,
                            whitespace=cst.SimpleWhitespace(value=''),
                            comment=comment.comment,
                            newline=cst.Newline(value=None)))
            break
        stmts.append(stmt)

    if not found_metadata:
        for stmt in func_node.body.footer:
            if stmt.comment is not None:
                if "# LITERALAI:" in stmt.comment.value:
                    found_metadata = True
                    break
                comments.append(stmt)

    return func_node.with_changes(
        body=func_node.body.with_changes(
            body = stmts,
            footer = comments
        ))

if __name__ == "__main__":
    source1 = """
def foo(x):
    "My docstring"
    # Some comment
    # LITERALAI: {"json": "goes", "here": "indeed"}
"""
    source2 = source1 + "\n    x = x + 1"

    for idx, source in enumerate([source1, source2]):
        module = cst.parse_module(source)
        func_node = module.body[0]

        data = extract_metadata(func_node)
        print("===={%s}====" % idx)
        print(data)

        #print(func_node)

        sig_node = extract_signature(func_node)
        print("Signature:")
        print(cst.Module([sig_node]).code_for_node(sig_node))
