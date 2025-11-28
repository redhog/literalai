import libcst as cst

def merge_fn(header: str, *bodies: str) -> str:
    """
    Merge a function header with multiple body sections using libcst.

    Args:
        header: Function definition including signature and docstring.
        *bodies: Arbitrary number of body sections as strings.

    Returns:
        Full function code as a string with proper indentation.
    """
    # Parse the header
    module = cst.parse_module(header)

    # Parse all body sections and combine their statements
    all_statements = []
    for body in bodies:
        body_module = cst.parse_module(body)
        all_statements.extend(body_module.body)

    # Transformer to replace the function body
    class BodyInserter(cst.CSTTransformer):
        def leave_FunctionDef(self, original_node, updated_node):
            return updated_node.with_changes(
                body=cst.IndentedBlock(body=all_statements)
            )

    # Apply the transformer
    new_module = module.visit(BodyInserter())

    return new_module.code


if __name__ == "__main__":
    # Example usage
    header = '''
    def add_two(a, b):
        """Add two numbers together, return the result

        Arguments:
        a -- first number
        b -- second number
        """
    '''

    body1 = '''
    c = a + b
    '''

    body2 = '''
    print("Sum:", c)
    '''

    body3 = '''
    return c
    '''

    merged_code = merge_fn(header, body1, body2, body3)
    print(merged_code)
