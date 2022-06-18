from setuptools import setup, find_packages

setup(
    name="example",
    version="0.1.0",
    author="Owen Bradley",
    author_email="owen.p.bradley@gmail.com",
    packages=find_packages(include=["example_package", "example_package.*"]),
    install_requires=["numpy", "scipy", "matplotlib"],
)
