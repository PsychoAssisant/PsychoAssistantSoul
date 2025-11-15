from setuptools import setup, find_packages

setup(
    name="PsychoAssistantSoul",              # nazwa pakietu
    version="0.3.0",                 # wersja pakietu
    packages=find_packages(),        # automatyczne znajdowanie modułów w folderze
    install_requires=[
        # core
        "transformers>=4.57.1,<5",
        "torch>=2.9.1,<3",

        # wymagane przez tokenizery T5 / niektóre modele HF
        "sentencepiece>=0.2.1",

        # biblioteki HF, które czasem warto jawnie zadeklarować
        "huggingface-hub>=0.36.0",
        "tokenizers>=0.22.1",

        # opcjonalne (przydatne jeśli używasz safetensors / szybszego ładowania modeli)
        "safetensors>=0.6.2"
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