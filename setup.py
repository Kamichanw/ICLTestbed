from typing import List
import os
from setuptools import setup, find_packages

ROOT_DIR = os.path.dirname(__file__)

def get_requirements(filename: str) -> List[str]:
    """Get Python package dependencies from a requirements file."""
    def _read_requirements(filename: str) -> List[str]:
        with open(os.path.join(ROOT_DIR, filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line.startswith("--"):
                continue
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    return _read_requirements(filename)

setup(
    name="testbed",
    version="0.1.0",
    description="An in-context learning research testbed",
    author="Kamichanw",
    author_email="865710157@qq.com",
    packages=find_packages(include=["testbed", "testbed.*"]),
    install_requires=get_requirements("requirements.txt"),
    extras_require={
        'dev': get_requirements("requirements-dev.txt"), 
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
