from setuptools import find_packages, setup

def get_requirements(file_path):
    with open(file_path) as file:
        requirements = file.read().splitlines()

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements

setup(
    name="Mlproject",
    version="0.0.1",
    author="Aditi Rao",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"))