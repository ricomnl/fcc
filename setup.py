"""Setup script for FCCC."""

from distutils.core import setup

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()


setup(
    name="fccc",
    version="0.1.0",
    description="Contact-based clustering for protein structures",
    long_description=readme,
    author="Joao Rodrigues",
    author_email="j.p.g.l.m.rodrigues@gmail.com",
    url="https://githuiub.com/joaorodrigues/fccc",
    license=license,
    py_modules=["fccc"],
    python_requires=">=3.9",
)
