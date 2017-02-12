from setuptools import find_packages
from setuptools import setup


setup(name='gandlf',
      version='0.0.9',
      description='Generative Adversarial Network Deep Learning Framework',
      author='Benjamin Bolte',
      author_email='bkbolte18@gmail.com',
      url='https://github.com/codekansas/gandlf',
      download_url='https://github.com/codekansas/gandlf/tarball/0.0.1',
      license='MIT',
      install_requires=['Keras==1.2.2'],
      packages=find_packages())
