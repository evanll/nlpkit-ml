import os
import setuptools

HERE = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()

with open(os.path.join(HERE, 'requirements.txt'), "r") as fp:
    install_reqs = fp.read().splitlines()

setuptools.setup(
    name="nlpkit-ml",
    version="1.0.0",
    author="Evan Lalopoulos",
    author_email="evan.lalopoulos.2017@my.bristol.ac.uk",
    description="A library of scikit compatible text transformers, that are ready to be integrated in an NLP pipeline "
                "for various classification tasks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evanll/nlpkit-ml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=install_reqs
)
