from setuptools import setup, find_packages

setup(
    name="PsychoAssistantSoul",
    version="0.4.0",
    packages=find_packages(),
    install_requires=[
        "openai==2.8.1",
        "transformers==4.57.1",
        "torch==2.9.1",
        "sentencepiece==0.2.1",
        "tokenizers==0.22.1",
        "huggingface-hub==0.36.0",
    ],
    author="Jacek Walczak",
    author_email="jacekwalczak4@gmail.com",
    description="SDK chatbota opartego na Hugging Face Transformers",
    url="https://github.com/twojanazwa/PsychoAssistantSoul",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)