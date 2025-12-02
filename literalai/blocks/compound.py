import libcst as cst
import json
from typing import Tuple, Dict, Optional

def unextract_metadata(data: dict) -> cst.BaseCompoundStatement:
    return append_body(
        signature_append_metadata(data["signature"], data["metadata"]),
        data["body"])

def extract_metadata(node: cst.BaseCompoundStatement) -> dict:
    """
    Extract JSON from a `# LITERALAI:` comment in a function body.

    Returns:
        - The parsed JSON as a dict, or {} if not found.
        - Boolean indicating whether the function has real statements.
    """

    literalai_data = {}
    signature = extract_signature(node)
    footer = []
    for stmt in signature["signature"].body.footer:
        if stmt.comment and "LITERALAI: " in stmt.comment.value:
            json_text = stmt.comment.value.split("# LITERALAI:")[1].strip()
            literalai_data.update(json.loads(json_text))
            break
        footer.append(stmt)
        
    return {
        **signature,
        "metadata": literalai_data,
            "signature": signature["signature"].with_changes(
                body=signature["signature"].body.with_changes(
                    footer = footer)),
            "empty":  signature["body"] is None,
            }

def signature_append_metadata(node: cst.BaseCompoundStatement, metadata: dict):
    return node.with_changes(
        body = node.body.with_changes(
            footer = list(node.body.footer) + [
                cst.EmptyLine(
                    indent=True,
                    whitespace=cst.SimpleWhitespace(value=''),
                    comment=cst.Comment(value="# LITERALAI: " + json.dumps(metadata)),
                    newline=cst.Newline(value=None))]))

def extract_signature(node: cst.BaseCompoundStatement) -> dict:
    comments = []
    header = []
    body = []
    body_comments = []
    found_metadata = False

    stmt  = None
    stmts = iter(node.body.body)
    for stmt in stmts:
        if (    not isinstance(stmt, cst.Pass)
            and not (    isinstance(stmt, cst.SimpleStatementLine)
                     and len(stmt.body) == 1
                     and isinstance(stmt.body[0], cst.Expr)
                     and isinstance(stmt.body[0].value, cst.SimpleString))):
            if stmt.leading_lines:
                for comment in stmt.leading_lines:
                    if comment.comment:
                        comments.append(
                            cst.EmptyLine(
                                indent=True,
                                whitespace=cst.SimpleWhitespace(value=''),
                                comment=comment.comment,
                                newline=cst.Newline(value=None)))
                stmt = stmt.with_changes(leading_lines=[])
            body.append(stmt)
            for stmt in stmts:
                body.append(stmt)
            break
        header.append(stmt)

    if body:
        for stmt in node.body.footer:
            body_comments.append(stmt)
    else:
        for stmt in node.body.footer:
            comments.append(stmt)

    return {
        "signature": node.with_changes(
            body=node.body.with_changes(
                body = header,
                footer = comments
            )),
        "body": cst.IndentedBlock(body=body, footer=body_comments) if body or body_comments else None}

def append_2blocks(a: cst.IndentedBlock|None, b: cst.IndentedBlock|None) -> cst.IndentedBlock:
    if a is None: return b
    if b is None: return a
    stmts = b.body
    footer = b.footer
    if a.footer:
        if stmts:
            stmts[0] = stmts[0].with_changes(
                leading_lines = a.footer + list(stmts[0].leading_lines))
        else:
            footer = a.footer + footer
    return a.with_changes(
            body = a.body + stmts,
            footer = footer)
    
def append_blocks(*blocks: list[cst.IndentedBlock]) -> cst.IndentedBlock:
    if len(blocks) == 1:
        return blocks[0]
    return append_2blocks(blocks[0], append_blocks(*blocks[1:]))

def append_body(signature: cst.BaseCompoundStatement, body: cst.IndentedBlock) -> cst.BaseCompoundStatement:
    return signature.with_changes(
        body = append_blocks(signature.body, body))


if __name__ == "__main__":
    source1 = """
def foo(x):
    "My docstring"
    # Some comment
    # Other comment
    # LITERALAI: {"foo": "bar"}
"""
    source2 = source1 + "\n    x = x + 1\n    # End comment"

    for idx, source in enumerate([source1, source2]):
        module = cst.parse_module(source)
        func_node = module.body[0]

        data = extract_metadata(func_node)
        print("===={%s}====" % idx)
        print(data["metadata"])
        print("Signature:")
        print(cst.Module([data["signature"]]).code_for_node(data["signature"]))
        print("Body:")
        if data["body"] is None:
            print("None")
        else:
            print(cst.Module([data["body"]]).code_for_node(data["body"]))

    dst = """
def foo(x):
    "My docstring"
    # Some comment
"""
    src = """
def bar(x):
    return x + 1
    # Yeah, really do just inc(x)
"""
    
    src = cst.parse_module(src).body[0]
    dst = cst.parse_module(dst).body[0]

    signature = extract_signature(dst)["signature"]
    body = extract_signature(src)["body"]

    res = append_body(signature, body)

    print("dst with body from src:")
    print(cst.Module([res]).code_for_node(res))


    newsig = signature_append_metadata(src, {"Metadata": "added"})
    print("src with added metadata:")
    print(cst.Module([newsig]).code_for_node(newsig))
