<img src="logo.jpeg" align="right" width="200">

# LiteralAI compiler for Python

What is literal ai coding? It's the **opposite of agentic AI** enabled
IDEs like Cursor, where your prompts modify your entire codebase and
are then quickly forgotten, leading to **write-only throw-away code**.
Literal ai is a way to store your **prompts inside your project**, and
check them into git just like the rest of your code. It works more
like a compiler - **compiling your prompts into source code**.

## How it works
Any function that only has a docstring and/or initial comments, will
be generated when `literalai` is run on your project, and any function
that was generated this way, will be updated if you change the
signature, docstring and/or comments. The LLM prompt will include the
function signature, docstring and any initial comments. Literal ai
does not store any metadata or state outside of your sourcecode, and
the only state it stores is a hash of the
signature/docstring/comments, as an extra comment inside functions it
generate.

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
    # LITERALAI: {"prompt": "somehash"}
    result = a + b
    return result

def manual(x):
    "A manual function"
    return x + 1
```
Notice how the manual function stays untouched. If you where to change
the docstring of `add_two` and rerun `literalai .`, its body would be
regenerated using the new docstring.

## What about classes?

Any class that only has a docstring and/or initial comments, will be
generated when `literalai` is run on your project, and any class that
was generated this way, will be updated if you change the signature,
docstring and/or comments.


What it means to update a class is that method signatures and
docstrings will be generated matching the docstring of the class.
These will then be generated as functions as per above. Any methods
previously generated this way and whos' docstrings or comments haven't
been manually changed, will be replaced by the new ones.

Methods that have manually written or changed signatures, docstrings
or initial comments won't be touched / overwritten.

## Models, system prompts and other configuration

Model names, system prompts / prompt templates and other configuration
is read from yaml files `literalai.yml` inside your project. All such
files found along the path from the project root to your source file
are merged and used as the config for that file.

Inside the config there are top level blocks: `base`, `FunctionDef`
and `ClassDef`. The two latter overrides the `base` section for
functions and classes respectively.

Each block can set the following keys:

* `model` - a `litellm` compatible model string such as `openai/gpt-4`
* `prompt` - a system prompt template processed using `jinja2` with
  access to the variable `signature` (the function or class signature)
  and all of the config.

Note: **Changes to the config** will lead to **code regeneration** for
affected code (e.g. `base` and `FunctionDef` regenerates functions).


Example `literalai.yml`:

```yaml
base:
  model: "openai/gpt-4"

FunctionDef:
  prompt: |
    Generate the python source code for a function with the following
    signature, docstring, and initial comments.

    {{signature}}

    # IMPORTANT
     * Write the full function implementation.
     * Provide only valid python for a single function as output.
     * Do NOT add any initial description, argument or similar

ClassDef:
  prompt: |
    Below is the python source code for a class and some of its methods
    (without implementations). Given the docstring and initial comments of
    the class, define any missing method signatures and provide their
    docstrings.

    {{signature}}

    # IMPORTANT
     * Write the full class specification.
     * Provide only valid skeleton python for a single class as output.
     * Do NOT add any initial description or similar
```

## Installation

`pip install literalai-python`
