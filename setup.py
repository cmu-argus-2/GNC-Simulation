from setuptools import setup, find_packages

setup(
    name='argus2sim',
    version='0.0.0',
    packages=find_packages(include=[
        'argus2sim',
        'argus2sim.*'
    ]),
    install_requires=[
        'numpy==1.26.4',
        'scipy==1.12.0',
        'matplotlib==3.8.3',
        'casadi==3.6.3',
    ]
)
