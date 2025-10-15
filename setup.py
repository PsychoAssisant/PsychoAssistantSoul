from setuptools import setup, find_packages

setup(
    name="PsychoAssistantSoul",              # nazwa pakietu
    version="0.1.0",                 # wersja pakietu
    packages=find_packages(),        # automatyczne znajdowanie modułów w folderze
    install_requires=[               # wymagane zależności
        "transformers>=4.0.0",
        "torch>=1.7.0"
    ],
    author="Jacek Walczak",             # autor pakietu
    author_email="jacekwalczak4@gmail.com",  # email autora (opcjonalnie)
    description="SDK chatbota opartego na Hugging Face Transformers",
    url="https://github.com/twojanazwa/PsychoAssistantSoul",  # adres repozytorium
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',         # minimalna wersja Pythona
)