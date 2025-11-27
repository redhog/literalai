import libcst as cst
from libcst import parse_module, Module, IndentedBlock

def set_indent(code: str, indent: str) -> str:
    """
    Indent all top-level statements in `code` with the given indent string,
    suitable for inserting into a function body.
    """
    module = parse_module(code)
    # Wrap all top-level statements in a single IndentedBlock
    block = IndentedBlock(body=module.body, indent=indent)
    
    # Create a dummy module containing just the indented block
    dummy_module = Module(body=[block])
    
    return dummy_module.code
