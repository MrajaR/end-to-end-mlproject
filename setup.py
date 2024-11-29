from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    with open(file_path) as f:
        requirements = f.read().splitlines()

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

        return requirements

setup(
    name='End2end-ML',
    packages=find_packages(),
    version='0.0.1',
    description='A portfolio of end-to-end ML projects',
    author='Raja Reivan',
    author_email='mrajareivan@gmail.com',
    install_requires=get_requirements('requirements.txt'))