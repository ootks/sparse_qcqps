import sys

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

from setuptools import find_packages

setup(
    name="lpm_methods",
    version="0.0.1",
    description="A library for performing sparse regression and sparse PCA.",
    author="Kevin Shu",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmake_install_dir="src/lpm_methods",
    include_package_data=True,
    python_requires=">=3.6",
)

