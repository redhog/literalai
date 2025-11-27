import libcst as cst
from typing import List, Tuple

def extract_fn_signature(func_def: cst.FunctionDef) -> Tuple[List[str], str]:
    """
    Extracts function signature, comments, docstring, and post-docstring comments
    from a libcst.FunctionDef node, preserving indentation and formatting.

    Returns:
        parts: List of strings [signature, leading_comments..., docstring, post_docstring_comments...]
        indentation: The string used for indentation of the function body
    """
    parts: List[str] = []

    # Temporary module to generate exact code snippets
    module = cst.Module([])

    # 1. Signature (including original formatting)
    full_func_code = module.code_for_node(func_def)
    signature_line = full_func_code.split(":", 1)[0] + ":"
    parts.append(signature_line)

    # 2. Leading comments (before the function, include indentation)
    for line in func_def.leading_lines:
        if line.comment:
            indent = line.indent.value if line.indent else ""
            parts.append(f"{indent}{line.comment.value}")

    # 3. Docstring (including quotes and indentation)
    docstring_node = None
    body_statements = func_def.body.body
    if body_statements:
        first_stmt = body_statements[0]
        if isinstance(first_stmt, cst.SimpleStatementLine) and first_stmt.body:
            expr = first_stmt.body[0]
            if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.SimpleString):
                docstring_node = expr.value
                # include indentation before docstring
                indent = first_stmt.leading_lines[0].indent.value if first_stmt.leading_lines else " " * 4
                docstring_code = module.code_for_node(docstring_node)
                docstring_lines = docstring_code.splitlines()
                docstring_with_indent = "\n".join([indent + line if line.strip() else line for line in docstring_lines])
                parts.append(docstring_with_indent)

    # 4. Comments after docstring but before real code
    post_doc_comments = []
    start_idx = 1 if docstring_node else 0
    for stmt in body_statements[start_idx:]:
        if isinstance(stmt, cst.SimpleStatementLine):
            for line in stmt.leading_lines:
                if line.comment:
                    indent = line.indent.value if line.indent else ""
                    post_doc_comments.append(f"{indent}{line.comment.value}")
            # Stop at first non-pass statement
            if not all(isinstance(s, cst.Pass) for s in stmt.body):
                break
    parts.extend(post_doc_comments)

    # 5. Indentation for function body (detect from first real statement)
    indentation = " " * 4  # default
    if body_statements:
        first_stmt = body_statements[0]
        if first_stmt.leading_lines:
            for line in first_stmt.leading_lines:
                if line.indent:
                    indentation = line.indent.value
                    break

    return parts, indentation
