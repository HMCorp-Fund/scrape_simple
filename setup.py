from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scrape_simple",
    version="0.1.3",
    author="Anton Pavlenko",
    author_email="apavlenko@hmcorp.fund",
    description="A web scraper that uses Tor for anonymity and supports media extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HMCorp-Fund/scrape_simple",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "requests>=2.28.0",
        "PySocks>=1.7.1",
        "stem>=1.8.0",
        "beautifulsoup4>=4.11.0",
        "lxml>=4.9.0",
        "tqdm>=4.66.0"
    ],
    extras_require={
        "russian": ["natasha>=1.6.0"],
        "ai": ["transformers>=4.25.0", "pillow>=9.0.0", "torch>=2.0.0"]
    },
    entry_points={
        "console_scripts": [
            "scrape-simple=scrape_simple.cli:main",
        ],
    },
)