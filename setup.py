import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyDiffGame",
    version="0.0.1",
    author="Dr. Aviran Sadon, Joshua Shay Kricheli and Prof. Gera Weiss",
    author_email="skricheli2@gmail.com",
    description="PyDiffGame is a Python implementation of a Nash Equilibrium solution to Differential Games, based on a reduction of Game Hamilton-Bellman-Jacobi (GHJB) equations to Game Algebraic and Differential Riccati equations, associated with Multi-Objective Dynamical Control Systems",
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
    python_requires=">=3.7",
)