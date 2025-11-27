from setuptools import setup, find_packages

setup(
    name="literalai",
    version="0.0.1",
    description="A LiteralAI compiler for Python",
    author="Egil Moeller",
    author_email="redhog@redhog.org",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "litellm>=1.0.0",
        "libcst"
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "literalai=literalai:main",
        ]
    },
)
