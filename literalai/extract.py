import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider
from typing import List, Optional, Tuple


class ExtractImportsFn(cst.CSTVisitor):
    def __init__(self):
        self.imports: List[cst.CSTNode] = []
        self.function_node: Optional[cst.FunctionDef] = None

    def ensure_import_newline(self, node: cst.CSTNode) -> cst.CSTNode:
        # Only modify Import / ImportFrom nodes
        if isinstance(node, (cst.Import, cst.ImportFrom)):
            return cst.SimpleStatementLine(
                body=[node],
                trailing_whitespace=cst.TrailingWhitespace(
                    whitespace=cst.SimpleWhitespace(""),  # spaces before comment
                    comment=None,
                    newline=cst.Newline()                  # the newline after the statement
                )
            )
        return node
        
    def visit_Import(self, node: cst.Import) -> None:
        self.imports.append(self.ensure_import_newline(node))

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        self.imports.append(self.ensure_import_newline(node))

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.function_node = node

    @classmethod
    def run(cls, code: str):
        module = cst.parse_module(code)
        extractor = cls()
        module.visit(extractor)
        return {"fn": extractor.function_node, "imports": extractor.imports}


def extract_fn_signature(func_def: cst.FunctionDef) -> Tuple[List[str], str]:
    """
    Extract lines from function start up to the first real statement,
    skipping Pass and docstrings.
    """

    # Get full source of the function
    source = cst.Module([func_def]).code_for_node(func_def)

    # Get full source of the function
    source_lines = source.splitlines()

    # Wrap the module for metadata
    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module)
    pos_map = wrapper.resolve(PositionProvider)
    func_node = wrapper.module.body[0]

    # Find first real statement
    first_real_stmt = None
    for stmt in func_node.body.body:
        # Skip Pass
        if isinstance(stmt, cst.Pass):
            continue
        # Skip docstring
        if (
            isinstance(stmt, cst.SimpleStatementLine)
            and len(stmt.body) == 1
            and isinstance(stmt.body[0], cst.Expr)
            and isinstance(stmt.body[0].value, cst.SimpleString)
        ):
            continue
        first_real_stmt = stmt
        break

    # Determine start line
    start_line = None
    if first_real_stmt:
        # Try metadata on the node itself
        if first_real_stmt in pos_map:
            start_line = pos_map[first_real_stmt].start.line
        else:
            # fallback: first child in metadata
            for child in first_real_stmt.children:
                if child in pos_map:
                    start_line = pos_map[child].start.line
                    break
        # final fallback: assume first line after def
        if start_line is None:
            start_line = 2  # relative to code_for_node
    else:
        start_line = len(source_lines) + 1  # no real statements

    # Slice lines up to first real statement
    parts = source_lines[: start_line - 1]
    body = source_lines[start_line - 1:]

    # Remove any empty lines after the last comment/docstring
    while parts and not parts[-1].strip():
        parts = parts[:-1]
    
    # Determine indentation
    if first_real_stmt and start_line <= len(source_lines):
        line = source_lines[start_line - 1]
        stripped = line.lstrip()
        indentation = line[: len(line) - len(stripped)]
    else:
        indentation = " " * 4
    
    return parts, body, indentation

# Example
if __name__ == "__main__":
    code = """
def test(x: int) -> str:
    # Docstrings are important
    '''My docstring'''
    # Some other comment
    
    return str(x + 1)
"""
    module = cst.parse_module(code)
    func_node = module.body[0]
    header, body, indent = extract_fn_signature(func_node)
    print("Indent:", repr(indent))
    print("===={Header}=====")
    for part in header:
        print(repr(part))
    print("===={Body}=====")
    for part in body:
        print(repr(part))
