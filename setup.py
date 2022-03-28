"""Setup script for FCCC."""

from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()


setup(
    name="fccpy",
    version="0.1.0",
    description="Contact-based clustering for protein structures",
    long_description=readme,
    author="Joao Rodrigues",
    author_email="j.p.g.l.m.rodrigues@gmail.com",
    url="https://github.com/joaorodrigues/fccpy",
    license=license,
    python_requires=">=3.8",
    install_requires=[
        "numba",
        "numpy",
        "h5py",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": ["fccpy=fccpy.cli:main"],
    },
)
