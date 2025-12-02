import libcst as cst
from typing import List, Optional, Tuple, Union

def extract_imports_and_def(module: cst.Module) -> dict:
    """
    Given a libcst.Module, return (imports, first top-level function/class definition).

    - imports: list of Import or ImportFrom nodes found before the first definition,
               with leading comments removed.
    - definition: the first FunctionDef or ClassDef node.
    
    Pass statements and docstrings before the first definition are ignored.
    """
    imports = []
    definition = None

    for stmt in module.body:
        if isinstance(stmt, (cst.FunctionDef, cst.ClassDef)):
            definition = stmt
            break

        if isinstance(stmt, cst.SimpleStatementLine):
            # If it's an import, collect it
            for element in stmt.body:
                if isinstance(element, (cst.Import, cst.ImportFrom)):
                    imports.append(
                        cst.SimpleStatementLine(
                            body=[element],
                            leading_lines=[],  # remove comments
                            trailing_whitespace=stmt.trailing_whitespace
                        )
                    )
                    break
            else:
                # Not an import, but maybe a pass or docstring → ignore
                if all(isinstance(e, (cst.Pass, cst.Expr)) for e in stmt.body):
                    continue
                else:
                    # Something else → stop scanning
                    break
        else:
            # Other statements before the definition → stop scanning
            break
    return {"imports": imports, "definition": definition}
