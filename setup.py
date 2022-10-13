import os.path as path
from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# parse the package version number
with open(path.join(path.dirname(__file__), 'tomotok/core/VERSION')) as version_file:
    version = version_file.read().strip()

setup(
    name='tomotok',
    version=version,
    license='EUPL 1.2',
    author='Jakub Svoboda, Jordan Cavalier, Ondrej Ficker, Martin Imrisek et al.',
    # author_email='svoboda@ipp.cas.cz',
    description='Collection of algorithms used for tokamak plasma tomography',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # url='',
    packages=find_packages(),
    namespace_packages=['tomotok'],
    include_package_data=True,
    python_requires='>=3.5',
    install_requires=['numpy>=1.13.3', 'scipy>=1.1.0', 'matplotlib>=2.2.2', 'h5py>=2.7.1'],
)
