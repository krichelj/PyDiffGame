import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyDiffGame",
    version="0.0.1",
    author="Dr. Aviran Sadon, Joshua Shay Kricheli and Prof. Gera Weiss",
    author_email="skricheli2@gmail.com",
    description="Multi-Objective Control Systems Simulator based on Differential Games",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/krichelj/PyDiffGame",
    project_urls={
        "Bug Tracker": "https://github.com/krichelj/PyDiffGame/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)