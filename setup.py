from setuptools import setup, find_packages

setup(
    name="AtomicOrbit",
    version="1.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "plotly",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "atomicorbit=atomicorbit.main:main",
        ],
    },
)