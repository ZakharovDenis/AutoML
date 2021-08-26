import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AutoML",
    version="0.0.1",
    author="Zakharov Denis",
    author_email="zakharov.denis22@gmail.com",
    description="Light AutoML tool for classic machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    install_requires=[
        'scikit-learn',
        'numpy',
        'pandas'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    package_data={'': ['*.csv']},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
