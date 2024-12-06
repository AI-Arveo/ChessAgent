# Create a basic setup.py file for the project

from setuptools import setup

setup(
    name="chess_project",
    version="0.1",
    description="5AI Chess Project",
    install_requires=[
        'python-chess',
    ]
)
