from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="literalai_python",
    version="0.0.4",
    description="A LiteralAI compiler for Python",
    author="Egil Moeller",
    author_email="redhog@redhog.org",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "litellm>=1.0.0",
        "libcst",
        "pyyaml"
        "jinja2"
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "literalai=literalai:main",
        ]
    },

    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/redhog/literalai"
)
