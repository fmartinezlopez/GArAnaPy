from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='garanapy',
    version='0.1.0',
    description='Simple Python module to analyze data from ND-GAr',
    long_description=readme,
    author='Francisco Martínez López',
    author_email='f.martinezlopez@qmul.ac.uk',
    url='https://github.com/fmartinezlopez/GArAnaPy',
    license=license,
    packages=find_packages(exclude=('scripts'))
)