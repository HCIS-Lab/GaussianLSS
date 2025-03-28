from setuptools import setup, find_packages
__version__ = '0.0.1'

setup(
    name='GaussianLSS',
    version=__version__,
    author='Shu-Wei Lu',
    author_email='shuweilu@nycu.edu.tw',
    url='https://github.com/HCIS-Lab/GaussianLSS',
    license='MIT',
    packages=find_packages(include=['GaussianLSS', 'GaussianLSS.*']),
    zip_safe=False,
)
