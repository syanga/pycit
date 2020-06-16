import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pycit",
    version="0.0.7",
    author="Alan Yang",
    author_email="ayang1097@gmail.com",
    description="Conditional independence testing and Markov blanket feature selection in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/syanga/pycit",
    packages=setuptools.find_packages(),
    install_requires=['scipy>=1.4.1', 'numpy>=1.17.4', 'scikit-learn>=0.22.1'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.6',
)
