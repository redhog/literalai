<img arc="logo.jpeg" float="right">

# LiteralAI for Python

What is literal ai coding? It's the opposite of agentic AI enabled IDEs like Cursor, where your prompts modify your entire codebase and are then quickly forgotten, leading to write-only throw-away code.
Literal ai is a way to store your prompts *inside* your project, and check them into git just like the rest of your code. It works more like a compiler - compiling your prompts into source code.

## How it works
Any function that only has a docstring and/or initial comments, will be generated when `literalai` is run on your project, and any function that was generated this way, will be updated if you change the signature, docstring and/or comments.
The LLM prompt will include the function signature, docstring and any initial comments. Literal ai does not store any metadata or state outside of your sourcecode, and the only state it stores is a hash of the signature/docstring/comments, as an extra comment inside functions it generate.

## Example

You write

```python
def add_two(a, b):
    """Add two numbers together, return the result"""

def manual(x):
    "A manual function"
    return x + 1
```

and run `literalai .`

The file will then be updated to contain

```python
def add_two(a, b):
    """Add two numbers together, return the result"""
    # CODEID:4a5c8e754c305b36907466707fbbcdc9883ba6499cddd35fb4e1923f7af4e2e4
    result = a + b
    return result

def manual(x):
    "A manual function"
    return x + 1
```
Notice how the manual function stays untouched. If you where to change the docstring of `add_two` and rerun `literalai .`, its body would be regenerated using the new docstring.
